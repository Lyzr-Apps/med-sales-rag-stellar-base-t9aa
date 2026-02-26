/**
 * Server-side RAG Knowledge Base API Route
 *
 * This route proxies requests to the Lyzr RAG API v3 (https://rag-prod.studio.lyzr.ai)
 * Full API spec: https://rag-prod.studio.lyzr.ai/docs
 *
 * CRITICAL API SPECIFICATIONS:
 *
 * 1. POST /api/rag (JSON body { ragId })  →  GET /v3/rag/documents/{rag_id}/
 *    - Content-Type: application/json
 *    - Lists documents in a knowledge base
 *    - Headers: x-api-key
 *
 * 2. POST /api/rag (formData with file)  →  POST /v3/train/{fileType}/?rag_id={id}
 *    - Content-Type: multipart/form-data
 *    - rag_id in QUERY parameter
 *    - fileType (pdf|docx|txt) in URL PATH
 *    - Body: multipart/form-data (file + parser params)
 *    - Headers: x-api-key
 *
 * 3. DELETE /api/rag (with JSON)  →  DELETE /v3/rag/{rag_id}/docs/
 *    - rag_id in URL PATH
 *    - Body: JSON array of filenames
 *    - Headers: x-api-key, Content-Type: application/json
 *
 * NEVER expose LYZR_API_KEY to client — always proxy through this route.
 */

import { NextRequest, NextResponse } from "next/server";

// CRITICAL: Remove the default 1MB body size limit for file uploads.
// Without this, Next.js App Router rejects any request body > 1MB,
// which silently breaks PDF/DOCX uploads.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const LYZR_RAG_BASE_URL = "https://rag-prod.studio.lyzr.ai/v3";
const LYZR_API_KEY = process.env.LYZR_API_KEY || "";

const FILE_TYPE_MAP: Record<string, "pdf" | "docx" | "txt"> = {
  "application/pdf": "pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    "docx",
  "text/plain": "txt",
};

// GET - Health check and document listing (for platform compatibility)
export async function GET(request: NextRequest) {
  try {
    if (!LYZR_API_KEY) {
      return NextResponse.json(
        { success: false, error: "LYZR_API_KEY not configured" },
        { status: 500 }
      );
    }

    // Check if ragId is provided as query parameter
    const ragId = request.nextUrl.searchParams.get("ragId");
    if (ragId) {
      const response = await fetch(
        `${LYZR_RAG_BASE_URL}/rag/documents/${encodeURIComponent(ragId)}/`,
        {
          method: "GET",
          headers: {
            accept: "application/json",
            "x-api-key": LYZR_API_KEY,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        const filePaths = Array.isArray(data)
          ? data
          : data.documents || data.data || [];

        const documents = filePaths.map((filePath: string) => {
          const fullPath =
            typeof filePath === "string" ? filePath : String(filePath);
          const fileName = fullPath.split("/").pop() || fullPath;
          const ext = fileName.split(".").pop()?.toLowerCase() || "";
          const fileType =
            ext === "pdf"
              ? "pdf"
              : ext === "docx"
                ? "docx"
                : ext === "txt"
                  ? "txt"
                  : "unknown";
          return { fileName, fullPath, fileType, status: "active" };
        });

        return NextResponse.json({
          success: true,
          documents,
          ragId,
          timestamp: new Date().toISOString(),
        });
      }
    }

    // Default: return API status
    return NextResponse.json({
      success: true,
      message: "RAG Knowledge Base API is running",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Server error",
      },
      { status: 500 }
    );
  }
}

// POST - List documents (JSON body) or Upload and train (formData)
export async function POST(request: NextRequest) {
  try {
    if (!LYZR_API_KEY) {
      return NextResponse.json(
        {
          success: false,
          error: "LYZR_API_KEY not configured on server",
        },
        { status: 500 }
      );
    }

    const contentType = request.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      // List documents flow (was GET)
      const body = await request.json();
      const { ragId } = body;

      if (!ragId) {
        return NextResponse.json(
          {
            success: false,
            error: "ragId is required",
          },
          { status: 400 }
        );
      }

      const response = await fetch(
        `${LYZR_RAG_BASE_URL}/rag/documents/${encodeURIComponent(ragId)}/`,
        {
          method: "GET",
          headers: {
            accept: "application/json",
            "x-api-key": LYZR_API_KEY,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        // Response is array of file paths like ["storage/voicestream-dev-guide.pdf"]
        const filePaths = Array.isArray(data)
          ? data
          : data.documents || data.data || [];

        const documents = filePaths.map((filePath: string) => {
          const fullPath = typeof filePath === 'string' ? filePath : String(filePath);
          const fileName = fullPath.split("/").pop() || fullPath;
          const ext = fileName.split(".").pop()?.toLowerCase() || "";
          const fileType =
            ext === "pdf"
              ? "pdf"
              : ext === "docx"
                ? "docx"
                : ext === "txt"
                  ? "txt"
                  : "unknown";

          return {
            fileName,
            fullPath,
            fileType,
            status: "active",
          };
        });

        return NextResponse.json({
          success: true,
          documents,
          ragId,
          timestamp: new Date().toISOString(),
        });
      } else {
        const errorText = await response.text();
        return NextResponse.json(
          {
            success: false,
            error: `Failed to get documents: ${response.status}`,
            details: errorText,
          },
          { status: response.status }
        );
      }
    } else {
      // Upload flow (formData)
      const formData = await request.formData();
      const ragId = formData.get("ragId") as string;
      const file = formData.get("file") as File;

      if (!ragId || !file) {
        return NextResponse.json(
          {
            success: false,
            error: "ragId and file are required",
          },
          { status: 400 }
        );
      }

      // Determine file type from MIME type first, then fall back to file extension
      let fileType = FILE_TYPE_MAP[file.type];
      if (!fileType) {
        const ext = file.name.split(".").pop()?.toLowerCase();
        if (ext === "pdf") fileType = "pdf";
        else if (ext === "docx") fileType = "docx";
        else if (ext === "txt") fileType = "txt";
      }
      if (!fileType) {
        return NextResponse.json(
          {
            success: false,
            error: `Unsupported file type: ${file.type || 'unknown'}. Supported: PDF, DOCX, TXT`,
          },
          { status: 400 }
        );
      }

      // Direct upload and train in one step
      // Try multiple parsers for best compatibility with all PDF types including
      // Claude artifacts, scanned documents, and PDFs with graphics/tables.
      // Larger chunk size (2000) preserves more context per chunk for better retrieval.
      const trainUrl = `${LYZR_RAG_BASE_URL}/train/${fileType}/?rag_id=${encodeURIComponent(ragId)}`;
      const trainHeaders = {
        "x-api-key": LYZR_API_KEY,
        accept: "application/json",
      };

      // Read file bytes once for reuse across parser attempts
      const fileBytes = await file.arrayBuffer();

      const buildTrainForm = (parser: string) => {
        const fd = new FormData();
        const blob = new Blob([fileBytes], { type: file.type || "application/octet-stream" });
        fd.append("file", blob, file.name);
        fd.append("data_parser", parser);
        fd.append("chunk_size", "2000");
        fd.append("chunk_overlap", "200");
        fd.append("extra_info", "{}");
        return fd;
      };

      // Try parsers in order: auto (best compat) → llmsherpa (layout-aware) → none (raw text)
      const parsers = ["auto", "llmsherpa", "none"];
      let trainData: any = null;
      let lastTrainError = "";

      for (const parser of parsers) {
        try {
          const resp = await fetch(trainUrl, {
            method: "POST",
            headers: trainHeaders,
            body: buildTrainForm(parser),
          });

          if (resp.ok) {
            trainData = await resp.json();
            break;
          }

          lastTrainError = await resp.text();
        } catch (e) {
          lastTrainError = e instanceof Error ? e.message : "Network error during upload";
        }
      }

      if (!trainData) {
        return NextResponse.json(
          {
            success: false,
            error: `Failed to train document after trying multiple parsers`,
            details: lastTrainError,
          },
          { status: 500 }
        );
      }

      return NextResponse.json({
        success: true,
        message: "Document uploaded and trained successfully",
        fileName: file.name,
        fileType,
        documentCount: trainData.document_count || trainData.chunks || 1,
        ragId,
        timestamp: new Date().toISOString(),
      });
    }
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Server error",
      },
      { status: 500 }
    );
  }
}

// PATCH - Crawl a website and add content to knowledge base
export async function PATCH(request: NextRequest) {
  try {
    if (!LYZR_API_KEY) {
      return NextResponse.json(
        {
          success: false,
          error: "LYZR_API_KEY not configured on server",
        },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { ragId, url } = body;

    if (!ragId || !url) {
      return NextResponse.json(
        {
          success: false,
          error: "ragId and url are required",
        },
        { status: 400 }
      );
    }

    const response = await fetch(`https://api.beta.architect.new/api/v1/rag/crawl`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": LYZR_API_KEY,
      },
      body: JSON.stringify({ url, rag_id: ragId }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        {
          success: false,
          error: `Failed to crawl website: ${response.status}`,
          details: errorText,
        },
        { status: response.status }
      );
    }

    return NextResponse.json({
      success: true,
      message:
        "Website crawl started successfully. Content will be available shortly.",
      url,
      ragId,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Server error",
      },
      { status: 500 }
    );
  }
}

// DELETE - Remove documents from knowledge base
export async function DELETE(request: NextRequest) {
  try {
    if (!LYZR_API_KEY) {
      return NextResponse.json(
        {
          success: false,
          error: "LYZR_API_KEY not configured on server",
        },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { ragId, documentNames } = body;

    if (!ragId || !documentNames || !Array.isArray(documentNames)) {
      return NextResponse.json(
        {
          success: false,
          error: "ragId and documentNames array are required",
        },
        { status: 400 }
      );
    }

    // First, try deleting with the provided names (which may include full paths)
    const response = await fetch(
      `${LYZR_RAG_BASE_URL}/rag/${encodeURIComponent(ragId)}/docs/`,
      {
        method: "DELETE",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
          "x-api-key": LYZR_API_KEY,
        },
        body: JSON.stringify(documentNames),
      }
    );

    if (response.ok) {
      return NextResponse.json({
        success: true,
        message: "Documents deleted successfully",
        deletedCount: documentNames.length,
        ragId,
        timestamp: new Date().toISOString(),
      });
    }

    // If first attempt failed, the API may need the full storage paths.
    // Fetch document list to find matching full paths and retry.
    const listResponse = await fetch(
      `${LYZR_RAG_BASE_URL}/rag/documents/${encodeURIComponent(ragId)}/`,
      {
        method: "GET",
        headers: {
          accept: "application/json",
          "x-api-key": LYZR_API_KEY,
        },
      }
    );

    if (listResponse.ok) {
      const listData = await listResponse.json();
      const allPaths: string[] = Array.isArray(listData)
        ? listData
        : listData.documents || listData.data || [];

      // Map provided filenames to their full storage paths
      const fullPaths = documentNames
        .map((name: string) => {
          // If the name already looks like a full path, use it directly
          if (name.includes("/")) return name;
          // Otherwise, find the matching full path from the listing
          return allPaths.find(
            (p: string) => p.endsWith(`/${name}`) || p === name
          );
        })
        .filter(Boolean) as string[];

      if (fullPaths.length > 0) {
        const retryResponse = await fetch(
          `${LYZR_RAG_BASE_URL}/rag/${encodeURIComponent(ragId)}/docs/`,
          {
            method: "DELETE",
            headers: {
              accept: "application/json",
              "Content-Type": "application/json",
              "x-api-key": LYZR_API_KEY,
            },
            body: JSON.stringify(fullPaths),
          }
        );

        if (retryResponse.ok) {
          return NextResponse.json({
            success: true,
            message: "Documents deleted successfully",
            deletedCount: fullPaths.length,
            ragId,
            timestamp: new Date().toISOString(),
          });
        }

        const retryErrorText = await retryResponse.text();
        return NextResponse.json(
          {
            success: false,
            error: `Failed to delete documents: ${retryResponse.status}`,
            details: retryErrorText,
          },
          { status: retryResponse.status }
        );
      }
    }

    const errorText = await response.text();
    return NextResponse.json(
      {
        success: false,
        error: `Failed to delete documents: ${response.status}`,
        details: errorText,
      },
      { status: response.status }
    );
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Server error",
      },
      { status: 500 }
    );
  }
}
