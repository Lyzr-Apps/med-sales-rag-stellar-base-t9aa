'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { callAIAgent, extractText } from '@/lib/aiAgent'
import { uploadAndTrainDocument, getDocuments, deleteDocuments } from '@/lib/ragKnowledgeBase'
import type { RAGDocument } from '@/lib/ragKnowledgeBase'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import {
  FiMessageSquare, FiClock, FiShield, FiDatabase, FiSend, FiRefreshCw,
  FiUsers, FiTrendingUp, FiActivity, FiPhone,
  FiDownload, FiUpload, FiTrash2, FiChevronLeft,
  FiSearch, FiCheck, FiAlertTriangle, FiMenu,
  FiFile, FiMaximize2
} from 'react-icons/fi'

// ========================
// CONSTANTS
// ========================
const MANAGER_AGENT_ID = '69a00df43dc260b752bd74d9'
const SALES_AGENT_ID = '69a00dc76fed800e9b9b52a9'
const HCP_AGENT_ID = '69a00dc7d43403a91b332805'
const COMPLIANCE_AGENT_ID = '69a00ddcfddac4fa01fac4e7'
const SALES_RAG_ID = '69a00d98f572c99c0ffb7691'
const HCP_RAG_ID = '69a00d9800c2d274880efd81'

const AGENTS_INFO = [
  { id: MANAGER_AGENT_ID, name: 'Query Orchestrator Manager', purpose: 'Routes queries and synthesizes responses' },
  { id: SALES_AGENT_ID, name: 'Sales & Territory Agent', purpose: 'Sales performance and regional data' },
  { id: HCP_AGENT_ID, name: 'HCP & Doctor Profile Agent', purpose: 'Doctor profiles and specialties' },
  { id: COMPLIANCE_AGENT_ID, name: 'Compliance Guard Agent', purpose: 'Ensures regulatory compliance' },
]

// ========================
// THEME
// ========================
const THEME_VARS = {
  '--background': '20 30% 4%',
  '--foreground': '35 20% 90%',
  '--card': '20 25% 7%',
  '--card-foreground': '35 20% 90%',
  '--popover': '20 25% 10%',
  '--popover-foreground': '35 20% 90%',
  '--primary': '35 20% 90%',
  '--primary-foreground': '20 30% 8%',
  '--secondary': '20 20% 12%',
  '--secondary-foreground': '35 20% 90%',
  '--accent': '36 60% 31%',
  '--accent-foreground': '35 20% 95%',
  '--destructive': '0 63% 31%',
  '--muted': '20 18% 15%',
  '--muted-foreground': '35 15% 55%',
  '--border': '20 18% 16%',
  '--input': '20 20% 20%',
  '--ring': '36 60% 31%',
  '--sidebar-background': '20 28% 6%',
  '--sidebar-foreground': '35 20% 90%',
  '--sidebar-border': '20 18% 12%',
  '--sidebar-primary': '36 60% 31%',
  '--sidebar-accent': '20 18% 12%',
  '--radius': '0.5rem',
} as React.CSSProperties

// ========================
// TYPES
// ========================
interface ParsedResponse {
  answer: string
  sources_consulted: string[]
  compliance_status: string
  domains_accessed: string[]
  confidence: string
  flags: string[]
}

interface ChatMessage {
  id: string
  role: 'user' | 'agent'
  content: string
  timestamp: string
  parsedResponse?: ParsedResponse
}

interface AuditEntry {
  id: string
  timestamp: string
  query: string
  responseStatus: string
  domainsAccessed: string[]
  confidence: string
  sessionId: string
  fullResponse: string
  sourcesConsulted: string[]
  flags: string[]
}

// ========================
// HELPERS
// ========================
function safeParseResult(result: any): any {
  if (!result) return {}
  if (typeof result === 'string') {
    try { return JSON.parse(result) } catch { return { answer: result } }
  }
  if (typeof result === 'object') {
    // Handle nested stringified JSON inside result fields
    if (typeof result.answer === 'string' && result.answer.startsWith('{')) {
      try {
        const inner = JSON.parse(result.answer)
        return { ...result, ...inner }
      } catch { /* keep original */ }
    }
    return result
  }
  return { answer: String(result) }
}

/**
 * Deep extraction: tries every known path the Lyzr API might return data through.
 * This handles manager-subagent patterns where responses can be nested multiple levels.
 */
function deepExtractResponse(apiResult: any): {
  answer: string
  sources_consulted: string[]
  compliance_status: string
  domains_accessed: string[]
  confidence: string
  flags: string[]
} {
  const defaults = {
    answer: '',
    sources_consulted: [] as string[],
    compliance_status: 'compliant',
    domains_accessed: [] as string[],
    confidence: 'medium',
    flags: [] as string[],
  }

  if (!apiResult) return defaults

  // Path 1: result.response.result (standard structured JSON)
  let data: any = null

  const responseObj = apiResult.response
  if (responseObj) {
    // Try result.response.result first
    if (responseObj.result) {
      data = safeParseResult(responseObj.result)
    }
    // Try result.response.message if result is empty
    if ((!data || !data.answer) && responseObj.message) {
      const msgParsed = safeParseResult(responseObj.message)
      if (msgParsed.answer) {
        data = msgParsed
      } else if (typeof responseObj.message === 'string' && responseObj.message.length > 0) {
        data = { ...defaults, answer: responseObj.message }
      }
    }
    // Try result.response.result.text, .response, .content, etc.
    if (data && !data.answer) {
      const textKeys = ['text', 'response', 'content', 'message', 'answer_text', 'summary', 'output']
      for (const key of textKeys) {
        if (data[key] && typeof data[key] === 'string') {
          data.answer = data[key]
          break
        }
      }
    }
  }

  // Path 2: Top-level fields on apiResult itself
  if (!data || !data.answer) {
    if (apiResult.answer) {
      data = safeParseResult(apiResult)
    } else if (apiResult.text) {
      data = { ...defaults, answer: apiResult.text }
    } else if (apiResult.message && typeof apiResult.message === 'string') {
      data = { ...defaults, answer: apiResult.message }
    }
  }

  // Path 3: raw_response fallback
  if ((!data || !data.answer) && apiResult.raw_response) {
    const rawParsed = safeParseResult(apiResult.raw_response)
    if (rawParsed.answer) {
      data = rawParsed
    } else if (typeof apiResult.raw_response === 'string') {
      data = { ...defaults, answer: apiResult.raw_response }
    }
  }

  if (!data) return defaults

  // Safely coerce arrays
  const safeArray = (val: any): string[] => {
    if (Array.isArray(val)) return val.map(String)
    if (typeof val === 'string') {
      try {
        const parsed = JSON.parse(val)
        if (Array.isArray(parsed)) return parsed.map(String)
      } catch { /* not an array string */ }
      return val.length > 0 ? [val] : []
    }
    return []
  }

  return {
    answer: data.answer || data.text || data.response || data.content || data.message || '',
    sources_consulted: safeArray(data.sources_consulted),
    compliance_status: (data.compliance_status || 'compliant').toLowerCase(),
    domains_accessed: safeArray(data.domains_accessed),
    confidence: (data.confidence || 'medium').toLowerCase(),
    flags: safeArray(data.flags),
  }
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

function formatFullTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toLocaleString()
  } catch {
    return ''
  }
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
}

// ========================
// MARKDOWN RENDERER
// ========================
function formatInline(text: string): React.ReactNode {
  // Handle bold, italic, inline code, and links
  const tokens: React.ReactNode[] = []
  let remaining = text
  let keyIdx = 0

  while (remaining.length > 0) {
    // Bold: **text**
    const boldMatch = remaining.match(/^(.*?)\*\*(.*?)\*\*(.*)$/s)
    if (boldMatch) {
      if (boldMatch[1]) tokens.push(<React.Fragment key={keyIdx++}>{boldMatch[1]}</React.Fragment>)
      tokens.push(<strong key={keyIdx++} className="font-semibold text-foreground">{boldMatch[2]}</strong>)
      remaining = boldMatch[3]
      continue
    }
    // Italic: *text*
    const italicMatch = remaining.match(/^(.*?)\*(.*?)\*(.*)$/s)
    if (italicMatch) {
      if (italicMatch[1]) tokens.push(<React.Fragment key={keyIdx++}>{italicMatch[1]}</React.Fragment>)
      tokens.push(<em key={keyIdx++}>{italicMatch[2]}</em>)
      remaining = italicMatch[3]
      continue
    }
    // Inline code: `text`
    const codeMatch = remaining.match(/^(.*?)`(.*?)`(.*)$/s)
    if (codeMatch) {
      if (codeMatch[1]) tokens.push(<React.Fragment key={keyIdx++}>{codeMatch[1]}</React.Fragment>)
      tokens.push(
        <code key={keyIdx++} className="px-1.5 py-0.5 rounded text-xs font-mono" style={{ backgroundColor: 'hsl(20 18% 15%)' }}>
          {codeMatch[2]}
        </code>
      )
      remaining = codeMatch[3]
      continue
    }
    // No more matches, push the rest
    tokens.push(<React.Fragment key={keyIdx++}>{remaining}</React.Fragment>)
    break
  }

  return tokens.length === 1 ? tokens[0] : <>{tokens}</>
}

function renderMarkdown(text: string) {
  if (!text) return null

  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  let inCodeBlock = false
  let codeLines: string[] = []
  let listItems: React.ReactNode[] = []
  let listType: 'ul' | 'ol' | null = null

  const flushList = (currentIdx: number) => {
    if (listItems.length > 0 && listType) {
      if (listType === 'ul') {
        elements.push(
          <ul key={`list-${currentIdx}`} className="ml-4 space-y-1 my-2 list-disc">
            {listItems}
          </ul>
        )
      } else {
        elements.push(
          <ol key={`list-${currentIdx}`} className="ml-4 space-y-1 my-2 list-decimal">
            {listItems}
          </ol>
        )
      }
      listItems = []
      listType = null
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // Code blocks
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={i} className="rounded-md p-3 text-xs font-mono overflow-x-auto my-2" style={{ backgroundColor: 'hsl(20 18% 12%)' }}>
            <code>{codeLines.join('\n')}</code>
          </pre>
        )
        codeLines = []
        inCodeBlock = false
      } else {
        flushList(i)
        inCodeBlock = true
      }
      continue
    }
    if (inCodeBlock) {
      codeLines.push(line)
      continue
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
      flushList(i)
      elements.push(<hr key={i} className="border-border my-3" />)
      continue
    }

    // Headers
    if (line.startsWith('#### ')) {
      flushList(i)
      elements.push(
        <h5 key={i} className="font-semibold text-sm mt-3 mb-1 tracking-wide" style={{ color: 'hsl(35 20% 85%)' }}>
          {formatInline(line.slice(5))}
        </h5>
      )
      continue
    }
    if (line.startsWith('### ')) {
      flushList(i)
      elements.push(
        <h4 key={i} className="font-semibold text-sm mt-4 mb-1.5 tracking-wide" style={{ color: 'hsl(36 60% 45%)' }}>
          {formatInline(line.slice(4))}
        </h4>
      )
      continue
    }
    if (line.startsWith('## ')) {
      flushList(i)
      elements.push(
        <h3 key={i} className="font-semibold text-base mt-4 mb-2 font-serif tracking-wide" style={{ color: 'hsl(36 60% 40%)' }}>
          {formatInline(line.slice(3))}
        </h3>
      )
      continue
    }
    if (line.startsWith('# ')) {
      flushList(i)
      elements.push(
        <h2 key={i} className="font-bold text-lg mt-5 mb-2 font-serif tracking-wide" style={{ color: 'hsl(36 60% 38%)' }}>
          {formatInline(line.slice(2))}
        </h2>
      )
      continue
    }

    // Blockquote
    if (line.startsWith('> ')) {
      flushList(i)
      elements.push(
        <blockquote key={i} className="border-l-2 pl-3 my-2 text-sm text-muted-foreground italic" style={{ borderColor: 'hsl(36 60% 31%)' }}>
          {formatInline(line.slice(2))}
        </blockquote>
      )
      continue
    }

    // Unordered list items
    if (line.match(/^(\s*)([-*+])\s(.+)/)) {
      const match = line.match(/^(\s*)([-*+])\s(.+)/)!
      const indent = match[1].length
      if (listType !== 'ul') {
        flushList(i)
        listType = 'ul'
      }
      listItems.push(
        <li key={`li-${i}`} className="text-sm leading-relaxed" style={{ marginLeft: indent > 0 ? `${indent * 8}px` : undefined }}>
          {formatInline(match[3])}
        </li>
      )
      continue
    }

    // Ordered list items
    if (/^\s*\d+\.\s/.test(line)) {
      const match = line.match(/^(\s*)\d+\.\s(.+)/)
      if (match) {
        const indent = match[1].length
        if (listType !== 'ol') {
          flushList(i)
          listType = 'ol'
        }
        listItems.push(
          <li key={`li-${i}`} className="text-sm leading-relaxed" style={{ marginLeft: indent > 0 ? `${indent * 8}px` : undefined }}>
            {formatInline(match[2])}
          </li>
        )
        continue
      }
    }

    // Empty line
    if (!line.trim()) {
      flushList(i)
      elements.push(<div key={i} className="h-2" />)
      continue
    }

    // Regular paragraph
    flushList(i)
    elements.push(
      <p key={i} className="text-sm leading-relaxed" style={{ lineHeight: '1.65', letterSpacing: '0.01em' }}>
        {formatInline(line)}
      </p>
    )
  }

  // Flush any remaining list
  flushList(lines.length)

  return <div className="space-y-1">{elements}</div>
}

// ========================
// SAMPLE DATA
// ========================
const SAMPLE_MESSAGES: ChatMessage[] = [
  {
    id: 'sample-1',
    role: 'user',
    content: 'Show me my assigned HCPs in the Northeast territory',
    timestamp: new Date(Date.now() - 300000).toISOString(),
  },
  {
    id: 'sample-2',
    role: 'agent',
    content: 'Based on your Northeast territory assignment, you have 12 active HCPs across 3 states.',
    timestamp: new Date(Date.now() - 295000).toISOString(),
    parsedResponse: {
      answer: 'Based on your Northeast territory assignment, you have **12 active HCPs** across 3 states.\n\n### Key Contacts\n- **Dr. Sarah Chen** - Cardiology, Mass General Hospital\n- **Dr. Michael Rivera** - Oncology, NYU Langone\n- **Dr. Emily Watson** - Neurology, Johns Hopkins\n- **Dr. James Park** - Internal Medicine, Yale New Haven\n\n### Territory Summary\n- **Massachusetts**: 4 HCPs (2 Tier 1, 2 Tier 2)\n- **New York**: 5 HCPs (3 Tier 1, 2 Tier 2)\n- **Connecticut**: 3 HCPs (1 Tier 1, 2 Tier 2)\n\nAll contacts are current and compliant with engagement policies.',
      sources_consulted: ['HCP Database', 'Territory Assignments', 'Compliance Registry'],
      compliance_status: 'compliant',
      domains_accessed: ['HCP Profiles', 'Territory Management'],
      confidence: 'high',
      flags: [],
    },
  },
  {
    id: 'sample-3',
    role: 'user',
    content: 'What was my Q2 regional performance?',
    timestamp: new Date(Date.now() - 200000).toISOString(),
  },
  {
    id: 'sample-4',
    role: 'agent',
    content: 'Your Q2 performance showed strong growth in the Northeast territory.',
    timestamp: new Date(Date.now() - 195000).toISOString(),
    parsedResponse: {
      answer: 'Your Q2 performance in the **Northeast territory** showed strong results:\n\n## Performance Overview\n- **Total Revenue**: $2.4M (up 18% vs Q1)\n- **Quota Attainment**: 112%\n- **New Prescriptions**: 847 (up 23%)\n- **HCP Meetings**: 48 completed\n\n### Top Performing Products\n1. **CardioMax 500mg** - $1.2M (142% of target)\n2. **NeuroShield XR** - $680K (98% of target)\n3. **OncoGuard Plus** - $520K (105% of target)\n\n### Areas for Improvement\n- Connecticut territory slightly below target at 89%\n- Follow-up meeting rate could improve (currently 72%)\n\nOverall ranking: **3rd out of 15 reps** nationally.',
      sources_consulted: ['Sales Analytics', 'Revenue Reports', 'Performance Dashboard'],
      compliance_status: 'compliant',
      domains_accessed: ['Sales Data', 'Performance Metrics'],
      confidence: 'high',
      flags: [],
    },
  },
]

const SAMPLE_AUDIT: AuditEntry[] = [
  {
    id: 'audit-1',
    timestamp: new Date(Date.now() - 300000).toISOString(),
    query: 'Show me my assigned HCPs in the Northeast territory',
    responseStatus: 'compliant',
    domainsAccessed: ['HCP Profiles', 'Territory Management'],
    confidence: 'high',
    sessionId: 'session-sample-001',
    fullResponse: 'Based on your Northeast territory assignment, you have 12 active HCPs across 3 states.',
    sourcesConsulted: ['HCP Database', 'Territory Assignments'],
    flags: [],
  },
  {
    id: 'audit-2',
    timestamp: new Date(Date.now() - 200000).toISOString(),
    query: 'What was my Q2 regional performance?',
    responseStatus: 'compliant',
    domainsAccessed: ['Sales Data', 'Performance Metrics'],
    confidence: 'high',
    sessionId: 'session-sample-001',
    fullResponse: 'Your Q2 performance in the Northeast territory showed strong results.',
    sourcesConsulted: ['Sales Analytics', 'Revenue Reports'],
    flags: [],
  },
  {
    id: 'audit-3',
    timestamp: new Date(Date.now() - 100000).toISOString(),
    query: 'Show me competitor pricing for CardioMax alternatives',
    responseStatus: 'redacted',
    domainsAccessed: ['Sales Data'],
    confidence: 'medium',
    sessionId: 'session-sample-001',
    fullResponse: 'Some competitor pricing details have been redacted per compliance policy.',
    sourcesConsulted: ['Sales Analytics'],
    flags: ['Competitor pricing data partially restricted'],
  },
]

// ========================
// SUGGESTED QUERIES
// ========================
const SUGGESTED_QUERIES = [
  { text: 'My assigned HCPs', icon: FiUsers, description: 'View your healthcare provider contacts' },
  { text: 'Q2 regional performance', icon: FiTrendingUp, description: 'Sales metrics and quota attainment' },
  { text: 'Doctor specialties in my territory', icon: FiActivity, description: 'Specialty distribution overview' },
  { text: 'Department contacts', icon: FiPhone, description: 'Key department contacts and details' },
]

// ========================
// NAV ITEMS
// ========================
type ViewType = 'chat' | 'history' | 'audit' | 'kb'

const NAV_ITEMS: { key: ViewType; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { key: 'chat', label: 'Chat', icon: FiMessageSquare },
  { key: 'history', label: 'Query History', icon: FiClock },
  { key: 'audit', label: 'Audit Log', icon: FiShield },
  { key: 'kb', label: 'Knowledge Base', icon: FiDatabase },
]

// ========================
// ERROR BOUNDARY
// ========================
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false, error: '' }
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
          <div className="text-center p-8 max-w-md">
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-muted-foreground mb-4 text-sm">{this.state.error}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: '' })}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm"
            >
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

// ========================
// COMPLIANCE BADGE
// ========================
function ComplianceBadge({ status }: { status: string }) {
  const s = (status ?? '').toLowerCase()
  if (s === 'compliant') {
    return (
      <Badge className="bg-green-900/50 text-green-300 border-green-700/50 hover:bg-green-900/50 text-xs">
        <FiCheck className="mr-1 h-3 w-3" /> Compliant
      </Badge>
    )
  }
  if (s === 'redacted') {
    return (
      <Badge className="bg-amber-900/50 text-amber-300 border-amber-700/50 hover:bg-amber-900/50 text-xs">
        <FiAlertTriangle className="mr-1 h-3 w-3" /> Redacted
      </Badge>
    )
  }
  if (s === 'flagged') {
    return (
      <Badge className="bg-red-900/50 text-red-300 border-red-700/50 hover:bg-red-900/50 text-xs">
        <FiAlertTriangle className="mr-1 h-3 w-3" /> Flagged
      </Badge>
    )
  }
  return <Badge variant="outline" className="text-xs">{status || 'Unknown'}</Badge>
}

// ========================
// CONFIDENCE INDICATOR
// ========================
function ConfidenceIndicator({ level }: { level: string }) {
  const l = (level ?? '').toLowerCase()
  const color = l === 'high' ? 'bg-green-500' : l === 'medium' ? 'bg-amber-500' : 'bg-red-500'
  const dots = l === 'high' ? 3 : l === 'medium' ? 2 : 1
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-muted-foreground">Confidence:</span>
      <div className="flex gap-0.5">
        {[1, 2, 3].map((d) => (
          <div key={d} className={`h-1.5 w-4 rounded-full ${d <= dots ? color : 'bg-muted'}`} />
        ))}
      </div>
      <span className="text-xs text-muted-foreground capitalize">{l || 'unknown'}</span>
    </div>
  )
}

// ========================
// LOADING SKELETON
// ========================
function AgentLoadingSkeleton() {
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[80%] w-full">
        <Card className="bg-card border-border shadow-lg">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
              <span className="text-xs text-muted-foreground animate-pulse">Retrieving from knowledge sources...</span>
            </div>
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-4/6" />
            <Skeleton className="h-3 w-3/6 mt-2" />
            <div className="flex gap-2 mt-3">
              <Skeleton className="h-5 w-20 rounded-full" />
              <Skeleton className="h-5 w-24 rounded-full" />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ========================
// KB UPLOAD SECTION
// ========================
function KBSection({
  title,
  ragId,
  docs,
  setDocs,
  loading,
  setLoading,
  statusMsg,
  setStatusMsg,
}: {
  title: string
  ragId: string
  docs: RAGDocument[]
  setDocs: React.Dispatch<React.SetStateAction<RAGDocument[]>>
  loading: boolean
  setLoading: React.Dispatch<React.SetStateAction<boolean>>
  statusMsg: string
  setStatusMsg: React.Dispatch<React.SetStateAction<string>>
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const loadDocs = useCallback(async () => {
    setLoading(true)
    setStatusMsg('')
    try {
      const result = await getDocuments(ragId)
      if (result.success && Array.isArray(result.documents)) {
        setDocs(result.documents)
      } else {
        setStatusMsg(result.error || 'Failed to load documents')
      }
    } catch {
      setStatusMsg('Error loading documents')
    }
    setLoading(false)
  }, [ragId, setDocs, setLoading, setStatusMsg])

  useEffect(() => {
    loadDocs()
  }, [loadDocs])

  const handleUpload = async (file: File) => {
    setLoading(true)
    setStatusMsg('')
    try {
      const result = await uploadAndTrainDocument(ragId, file)
      if (result.success) {
        setStatusMsg(`"${file.name}" uploaded successfully`)
        await loadDocs()
      } else {
        setStatusMsg(result.error || 'Upload failed')
      }
    } catch {
      setStatusMsg('Error uploading file')
    }
    setLoading(false)
  }

  const handleDelete = async (fileName: string) => {
    setLoading(true)
    setStatusMsg('')
    try {
      const result = await deleteDocuments(ragId, [fileName])
      if (result.success) {
        setStatusMsg(`"${fileName}" deleted`)
        await loadDocs()
      } else {
        setStatusMsg(result.error || 'Delete failed')
      }
    } catch {
      setStatusMsg('Error deleting document')
    }
    setLoading(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleUpload(file)
  }

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold flex items-center gap-2">
          <FiDatabase className="h-4 w-4 text-accent" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${dragOver ? 'border-accent bg-accent/10' : 'border-border hover:border-muted-foreground'}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <FiUpload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Drop files here or click to upload</p>
          <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, TXT supported</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) handleUpload(f)
              e.target.value = ''
            }}
          />
        </div>

        {statusMsg && (
          <p className={`text-xs ${statusMsg.includes('failed') || statusMsg.includes('Error') || statusMsg.includes('Failed') ? 'text-red-400' : 'text-green-400'}`}>
            {statusMsg}
          </p>
        )}

        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{docs.length} document{docs.length !== 1 ? 's' : ''}</span>
          <Button variant="ghost" size="sm" onClick={loadDocs} disabled={loading} className="h-7 text-xs">
            <FiRefreshCw className={`h-3 w-3 mr-1 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </Button>
        </div>

        {docs.length > 0 && (
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {docs.map((doc, idx) => (
              <div key={doc.id || doc.fileName || idx} className="flex items-center justify-between p-2 rounded-md bg-secondary/50 border border-border">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <FiFile className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <span className="text-xs truncate">{doc.fileName}</span>
                  {doc.status && (
                    <Badge variant="outline" className="text-[10px] flex-shrink-0">
                      {doc.status}
                    </Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive flex-shrink-0"
                  onClick={() => handleDelete(doc.fileName)}
                  disabled={loading}
                >
                  <FiTrash2 className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ========================
// MAIN PAGE
// ========================
export default function Page() {
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => generateId())

  // Navigation
  const [activeView, setActiveView] = useState<ViewType>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Audit log
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([])
  const [auditSearch, setAuditSearch] = useState('')
  const [auditFilter, setAuditFilter] = useState<string>('all')
  const [auditDetailEntry, setAuditDetailEntry] = useState<AuditEntry | null>(null)

  // Knowledge base
  const [salesDocs, setSalesDocs] = useState<RAGDocument[]>([])
  const [hcpDocs, setHcpDocs] = useState<RAGDocument[]>([])
  const [salesLoading, setSalesLoading] = useState(false)
  const [hcpLoading, setHcpLoading] = useState(false)
  const [salesStatus, setSalesStatus] = useState('')
  const [hcpStatus, setHcpStatus] = useState('')

  // History search
  const [historySearch, setHistorySearch] = useState('')

  // Sample data toggle
  const [showSample, setShowSample] = useState(false)

  // Active agent tracking
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null)

  // Refs
  const chatEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Get current display data
  const displayMessages = showSample ? SAMPLE_MESSAGES : messages
  const displayAudit = showSample ? SAMPLE_AUDIT : auditLog

  // ========================
  // SEND MESSAGE
  // ========================
  const handleSend = async (customMessage?: string) => {
    const msg = customMessage || inputValue.trim()
    if (!msg || isLoading) return

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: msg,
      timestamp: new Date().toISOString(),
    }

    setMessages(prev => [...prev, userMsg])
    setInputValue('')
    setIsLoading(true)
    setActiveAgentId(MANAGER_AGENT_ID)

    try {
      const result = await callAIAgent(msg, MANAGER_AGENT_ID, { session_id: sessionId })

      // Use deep extraction to handle all possible response structures
      const extracted = deepExtractResponse(result)

      // Fallback: if deep extraction found nothing, try extractText utility
      let finalAnswer = extracted.answer
      if (!finalAnswer && result.response) {
        finalAnswer = extractText(result.response)
      }
      if (!finalAnswer) {
        finalAnswer = 'I was unable to retrieve relevant information for your query. Please ensure knowledge base documents have been uploaded, or try rephrasing your question.'
      }

      const agentMsg: ChatMessage = {
        id: generateId(),
        role: 'agent',
        content: finalAnswer,
        timestamp: new Date().toISOString(),
        parsedResponse: {
          answer: finalAnswer,
          sources_consulted: extracted.sources_consulted,
          compliance_status: extracted.compliance_status,
          domains_accessed: extracted.domains_accessed,
          confidence: extracted.confidence,
          flags: extracted.flags,
        },
      }

      setMessages(prev => [...prev, agentMsg])

      const auditEntry: AuditEntry = {
        id: generateId(),
        timestamp: new Date().toISOString(),
        query: userMsg.content,
        responseStatus: extracted.compliance_status,
        domainsAccessed: extracted.domains_accessed,
        confidence: extracted.confidence,
        sessionId: sessionId,
        fullResponse: extracted.answer || finalAnswer,
        sourcesConsulted: extracted.sources_consulted,
        flags: extracted.flags,
      }
      setAuditLog(prev => [...prev, auditEntry])
    } catch (err) {
      const errorDetail = err instanceof Error ? err.message : 'Unknown error'
      const errorMsg: ChatMessage = {
        id: generateId(),
        role: 'agent',
        content: `An error occurred while processing your query: ${errorDetail}. Please try again.`,
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, errorMsg])
    }

    setIsLoading(false)
    setActiveAgentId(null)
  }

  // ========================
  // KEY HANDLER
  // ========================
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // ========================
  // NEW SESSION
  // ========================
  const handleNewSession = () => {
    setMessages([])
    setSessionId(generateId())
    setInputValue('')
  }

  // ========================
  // EXPORT AUDIT CSV
  // ========================
  const exportAuditCSV = () => {
    const rows = displayAudit.map(e => [
      formatFullTimestamp(e.timestamp),
      `"${e.query.replace(/"/g, '""')}"`,
      e.responseStatus,
      `"${(Array.isArray(e.domainsAccessed) ? e.domainsAccessed : []).join(', ')}"`,
      e.confidence,
      e.sessionId,
    ])
    const csv = 'Timestamp,Query,Status,Domains,Confidence,Session ID\n' + rows.map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `audit_log_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ========================
  // FILTERED AUDIT
  // ========================
  const filteredAudit = displayAudit.filter(e => {
    const matchSearch = !auditSearch || e.query.toLowerCase().includes(auditSearch.toLowerCase())
    const matchFilter = auditFilter === 'all' || e.responseStatus.toLowerCase() === auditFilter
    return matchSearch && matchFilter
  })

  // ========================
  // FILTERED HISTORY
  // ========================
  const filteredHistory = displayMessages.filter(m => {
    if (m.role !== 'user') return false
    if (!historySearch) return true
    return m.content.toLowerCase().includes(historySearch.toLowerCase())
  })

  // ========================
  // RENDER
  // ========================
  return (
    <ErrorBoundary>
      <div style={THEME_VARS} className="min-h-screen h-screen bg-background text-foreground flex overflow-hidden font-sans">
        {/* ==================== SIDEBAR ==================== */}
        <aside className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 flex-shrink-0 overflow-hidden`}>
          <div className="w-64 h-full flex flex-col border-r border-border" style={{ backgroundColor: 'hsl(20 28% 6%)' }}>
            {/* Branding */}
            <div className="p-5 border-b border-border">
              <h1 className="font-serif text-lg font-bold tracking-wide" style={{ color: 'hsl(36 60% 31%)' }}>
                MedRep
              </h1>
              <p className="text-xs text-muted-foreground mt-0.5 tracking-wide">Intelligence Hub</p>
            </div>

            {/* Nav items */}
            <nav className="flex-1 p-3 space-y-1">
              {NAV_ITEMS.map(item => {
                const Icon = item.icon
                const isActive = activeView === item.key
                return (
                  <button
                    key={item.key}
                    onClick={() => setActiveView(item.key)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-all duration-200 ${isActive ? 'text-accent-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'}`}
                    style={isActive ? { backgroundColor: 'hsl(20 18% 12%)', color: 'hsl(36 60% 31%)' } : undefined}
                  >
                    <Icon className="h-4 w-4 flex-shrink-0" />
                    <span className="tracking-wide">{item.label}</span>
                  </button>
                )
              })}
            </nav>

            {/* Agent Status */}
            <div className="p-3 border-t border-border">
              <p className="text-[10px] uppercase text-muted-foreground tracking-widest mb-2 px-2">Agents</p>
              <div className="space-y-1.5">
                {AGENTS_INFO.map(agent => (
                  <div key={agent.id} className="flex items-center gap-2 px-2 py-1">
                    <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${activeAgentId === agent.id ? 'bg-green-400 animate-pulse' : 'bg-muted-foreground/40'}`} />
                    <span className="text-[11px] text-muted-foreground truncate">{agent.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* ==================== MAIN ==================== */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* HEADER */}
          <header className="h-14 border-b border-border flex items-center justify-between px-4 flex-shrink-0 bg-card">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(!sidebarOpen)} className="h-8 w-8 p-0">
                {sidebarOpen ? <FiChevronLeft className="h-4 w-4" /> : <FiMenu className="h-4 w-4" />}
              </Button>
              <Separator orientation="vertical" className="h-6" />
              <span className="text-sm font-medium tracking-wide hidden sm:inline">MedRep Intelligence Hub</span>
            </div>

            <div className="flex items-center gap-3">
              {/* Sample data toggle */}
              <div className="flex items-center gap-2">
                <Label htmlFor="sample-toggle" className="text-xs text-muted-foreground">Sample Data</Label>
                <Switch
                  id="sample-toggle"
                  checked={showSample}
                  onCheckedChange={setShowSample}
                />
              </div>

              <Badge variant="outline" className="text-[10px] border-green-700/50 text-green-400 hidden md:flex">
                <FiCheck className="mr-1 h-3 w-3" />
                SOC2 | ISO27001 Compliant
              </Badge>

              <div className="h-8 w-8 rounded-full flex items-center justify-center text-xs font-medium" style={{ backgroundColor: 'hsl(36 60% 31%)', color: 'hsl(20 30% 8%)' }}>
                MR
              </div>
            </div>
          </header>

          {/* CONTENT AREA */}
          <div className="flex-1 overflow-hidden">
            {/* ==================== CHAT VIEW ==================== */}
            {activeView === 'chat' && (
              <div className="h-full flex flex-col">
                {/* Messages */}
                <ScrollArea className="flex-1 px-4 py-4">
                  {displayMessages.length === 0 && !isLoading ? (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                      <div className="text-center max-w-lg">
                        <h2 className="font-serif text-2xl font-bold mb-2 tracking-wide" style={{ color: 'hsl(36 60% 31%)' }}>
                          Welcome to MedRep Intelligence Hub
                        </h2>
                        <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                          Ask questions about your sales performance, assigned HCPs, regional data, and more. Your queries are routed through multiple knowledge domains and compliance-checked in real time.
                        </p>
                        <div className="mb-6 p-3 rounded-lg border border-border text-left" style={{ backgroundColor: 'hsl(20 25% 7%)' }}>
                          <div className="flex items-start gap-2.5">
                            <FiDatabase className="h-4 w-4 mt-0.5 flex-shrink-0" style={{ color: 'hsl(36 60% 45%)' }} />
                            <div>
                              <p className="text-xs font-medium mb-1" style={{ color: 'hsl(36 60% 50%)' }}>
                                Get started: Upload your documents
                              </p>
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                Upload sales reports, HCP profiles, territory data, and other documents to the Knowledge Base. The agent answers exclusively from your uploaded content -- including tables, charts, and structured data from PDFs.
                              </p>
                              <Button
                                variant="outline"
                                size="sm"
                                className="mt-2 h-7 text-xs"
                                style={{ borderColor: 'hsl(36 60% 31% / 0.4)', color: 'hsl(36 60% 50%)' }}
                                onClick={() => setActiveView('kb')}
                              >
                                <FiUpload className="h-3 w-3 mr-1.5" />
                                Upload Documents
                              </Button>
                            </div>
                          </div>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {SUGGESTED_QUERIES.map((sq) => {
                            const SqIcon = sq.icon
                            return (
                              <Card
                                key={sq.text}
                                className="bg-card border-border hover:border-accent/50 transition-all duration-200 cursor-pointer group"
                                onClick={() => handleSend(sq.text)}
                              >
                                <CardContent className="p-4 flex items-start gap-3">
                                  <div className="h-9 w-9 rounded-md flex items-center justify-center flex-shrink-0 transition-colors" style={{ backgroundColor: 'hsl(20 18% 12%)' }}>
                                    <SqIcon className="h-4 w-4" style={{ color: 'hsl(36 60% 31%)' }} />
                                  </div>
                                  <div className="text-left min-w-0">
                                    <p className="text-sm font-medium group-hover:text-foreground">{sq.text}</p>
                                    <p className="text-xs text-muted-foreground mt-0.5">{sq.description}</p>
                                  </div>
                                </CardContent>
                              </Card>
                            )
                          })}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-3xl mx-auto space-y-4 pb-4">
                      {displayMessages.map((msg) => {
                        if (msg.role === 'user') {
                          return (
                            <div key={msg.id} className="flex justify-end">
                              <div className="max-w-[75%]">
                                <div className="rounded-xl px-4 py-3 bg-secondary border border-border">
                                  <p className="text-sm">{msg.content}</p>
                                </div>
                                <p className="text-[10px] text-muted-foreground mt-1 text-right">{formatTimestamp(msg.timestamp)}</p>
                              </div>
                            </div>
                          )
                        }

                        const pr = msg.parsedResponse
                        const hasMetadata = pr && (
                          (Array.isArray(pr.domains_accessed) && pr.domains_accessed.length > 0) ||
                          (Array.isArray(pr.sources_consulted) && pr.sources_consulted.length > 0) ||
                          pr.compliance_status ||
                          (Array.isArray(pr.flags) && pr.flags.length > 0)
                        )
                        // Detect if response indicates no documents found
                        const answerText = (pr?.answer || msg.content || '').toLowerCase()
                        const isNoDocsResponse = pr?.confidence === 'low' ||
                          answerText.includes('not found in the uploaded') ||
                          answerText.includes('not found in the knowledge') ||
                          answerText.includes('no relevant') ||
                          answerText.includes('please upload') ||
                          answerText.includes('knowledge base is empty') ||
                          answerText.includes('no documents')

                        return (
                          <div key={msg.id} className="flex justify-start">
                            <div className="max-w-[85%] w-full">
                              <Card className="bg-card border-border shadow-lg">
                                <CardContent className="p-5">
                                  {/* Agent label */}
                                  <div className="flex items-center gap-2 mb-3">
                                    <div className="h-2 w-2 rounded-full flex-shrink-0" style={{ backgroundColor: 'hsl(36 60% 31%)' }} />
                                    <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Intelligence Hub</span>
                                    {pr && <ComplianceBadge status={pr.compliance_status} />}
                                  </div>

                                  {/* Answer content */}
                                  <div className="mb-4">
                                    {renderMarkdown(pr?.answer || msg.content)}
                                  </div>

                                  {/* Upload prompt when no documents found */}
                                  {isNoDocsResponse && (
                                    <div className="mb-4 p-3 rounded-lg border border-amber-700/30" style={{ backgroundColor: 'hsl(36 60% 31% / 0.08)' }}>
                                      <div className="flex items-start gap-2.5">
                                        <FiDatabase className="h-4 w-4 mt-0.5 flex-shrink-0" style={{ color: 'hsl(36 60% 45%)' }} />
                                        <div>
                                          <p className="text-xs font-medium mb-1" style={{ color: 'hsl(36 60% 50%)' }}>
                                            Knowledge Base documents needed
                                          </p>
                                          <p className="text-xs text-muted-foreground leading-relaxed">
                                            Upload relevant PDF, DOCX, or TXT files to the Knowledge Base so the agent can answer from your actual data. Documents with tables, charts, and structured data are supported.
                                          </p>
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            className="mt-2 h-7 text-xs border-amber-700/40 hover:border-amber-600/60"
                                            style={{ color: 'hsl(36 60% 50%)' }}
                                            onClick={() => setActiveView('kb')}
                                          >
                                            <FiUpload className="h-3 w-3 mr-1.5" />
                                            Go to Knowledge Base
                                          </Button>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Metadata footer */}
                                  {hasMetadata && (
                                    <div className="space-y-2.5 pt-3 border-t border-border">
                                      {/* Domains accessed row */}
                                      {Array.isArray(pr.domains_accessed) && pr.domains_accessed.length > 0 && (
                                        <div className="flex flex-wrap items-center gap-1.5">
                                          <span className="text-[10px] text-muted-foreground uppercase tracking-wider mr-1">Domains:</span>
                                          {pr.domains_accessed.map((d, i) => (
                                            <Badge key={i} className="text-[10px] px-2 py-0.5 border-0 font-medium" style={{ backgroundColor: 'hsl(36 60% 31% / 0.2)', color: 'hsl(36 60% 50%)' }}>
                                              {d}
                                            </Badge>
                                          ))}
                                        </div>
                                      )}

                                      {/* Sources consulted row */}
                                      {Array.isArray(pr.sources_consulted) && pr.sources_consulted.length > 0 && (
                                        <div className="flex flex-wrap items-center gap-1.5">
                                          <span className="text-[10px] text-muted-foreground uppercase tracking-wider mr-1">Documents:</span>
                                          {pr.sources_consulted.map((s, i) => (
                                            <Badge key={i} variant="outline" className="text-[10px] px-2 py-0.5">
                                              <FiFile className="h-2.5 w-2.5 mr-1" />{s}
                                            </Badge>
                                          ))}
                                        </div>
                                      )}

                                      {/* Confidence row */}
                                      <div className="flex flex-wrap items-center gap-3">
                                        <ConfidenceIndicator level={pr.confidence} />
                                      </div>

                                      {/* Flags row */}
                                      {Array.isArray(pr.flags) && pr.flags.length > 0 && (
                                        <div className="flex flex-wrap gap-1.5">
                                          {pr.flags.map((flag, i) => (
                                            <Badge key={i} className="text-[10px] bg-amber-900/30 text-amber-300 border-amber-700/30 hover:bg-amber-900/30">
                                              <FiAlertTriangle className="h-2.5 w-2.5 mr-1" /> {flag}
                                            </Badge>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </CardContent>
                              </Card>
                              <p className="text-[10px] text-muted-foreground mt-1">{formatTimestamp(msg.timestamp)}</p>
                            </div>
                          </div>
                        )
                      })}

                      {isLoading && <AgentLoadingSkeleton />}
                      <div ref={chatEndRef} />
                    </div>
                  )}
                </ScrollArea>

                {/* Input Bar */}
                <div className="border-t border-border p-4 bg-card flex-shrink-0">
                  <div className="max-w-3xl mx-auto">
                    <div className="flex gap-2 items-end">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleNewSession}
                        disabled={isLoading}
                        className="h-10 w-10 p-0 flex-shrink-0 text-muted-foreground hover:text-foreground"
                        title="New session"
                      >
                        <FiRefreshCw className="h-4 w-4" />
                      </Button>
                      <div className="flex-1 relative">
                        <Textarea
                          ref={textareaRef}
                          value={inputValue}
                          onChange={(e) => setInputValue(e.target.value)}
                          onKeyDown={handleKeyDown}
                          placeholder="Ask about sales, regions, HCPs, doctors..."
                          disabled={isLoading || showSample}
                          className="min-h-[44px] max-h-32 resize-none pr-12 bg-input border-border text-sm"
                          rows={1}
                        />
                        <Button
                          size="sm"
                          onClick={() => handleSend()}
                          disabled={!inputValue.trim() || isLoading || showSample}
                          className="absolute right-2 bottom-2 h-7 w-7 p-0"
                          style={{ backgroundColor: 'hsl(36 60% 31%)', color: 'hsl(20 30% 8%)' }}
                        >
                          <FiSend className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2 text-center">
                      Queries are routed through compliance-checked knowledge domains. Press Enter to send, Shift+Enter for newline.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* ==================== HISTORY VIEW ==================== */}
            {activeView === 'history' && (
              <div className="h-full flex flex-col p-4">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-serif text-lg font-semibold tracking-wide" style={{ color: 'hsl(36 60% 31%)' }}>
                    Query History
                  </h2>
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <FiSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                      <Input
                        placeholder="Search queries..."
                        value={historySearch}
                        onChange={(e) => setHistorySearch(e.target.value)}
                        className="h-8 pl-8 text-xs w-48 bg-input border-border"
                      />
                    </div>
                  </div>
                </div>

                <ScrollArea className="flex-1">
                  {filteredHistory.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-center">
                      <FiClock className="h-10 w-10 text-muted-foreground/30 mb-3" />
                      <p className="text-sm text-muted-foreground">
                        {showSample ? 'No matching queries' : 'No queries yet. Start a conversation to see history here.'}
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2 max-w-2xl">
                      <p className="text-xs text-muted-foreground mb-2">Session: {sessionId.substring(0, 8)}...</p>
                      {filteredHistory.map((userMsg) => {
                        const responseMsg = displayMessages.find(
                          m => m.role === 'agent' && displayMessages.indexOf(m) === displayMessages.indexOf(userMsg) + 1
                        )
                        const pr = responseMsg?.parsedResponse
                        return (
                          <Card
                            key={userMsg.id}
                            className="bg-card border-border hover:border-accent/30 transition-colors cursor-pointer"
                            onClick={() => setActiveView('chat')}
                          >
                            <CardContent className="p-3">
                              <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0 flex-1">
                                  <p className="text-sm font-medium truncate">{userMsg.content}</p>
                                  <p className="text-[10px] text-muted-foreground mt-1">{formatFullTimestamp(userMsg.timestamp)}</p>
                                </div>
                                <div className="flex flex-col items-end gap-1 flex-shrink-0">
                                  {pr && <ComplianceBadge status={pr.compliance_status} />}
                                </div>
                              </div>
                              {pr && Array.isArray(pr.domains_accessed) && pr.domains_accessed.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                  {pr.domains_accessed.map((d, i) => (
                                    <Badge key={i} variant="outline" className="text-[10px] px-1.5 py-0">{d}</Badge>
                                  ))}
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        )
                      })}
                    </div>
                  )}
                </ScrollArea>
              </div>
            )}

            {/* ==================== AUDIT LOG VIEW ==================== */}
            {activeView === 'audit' && (
              <div className="h-full flex flex-col p-4">
                <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
                  <h2 className="font-serif text-lg font-semibold tracking-wide" style={{ color: 'hsl(36 60% 31%)' }}>
                    Audit Log
                  </h2>
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <FiSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                      <Input
                        placeholder="Search..."
                        value={auditSearch}
                        onChange={(e) => setAuditSearch(e.target.value)}
                        className="h-8 pl-8 text-xs w-40 bg-input border-border"
                      />
                    </div>
                    <select
                      value={auditFilter}
                      onChange={(e) => setAuditFilter(e.target.value)}
                      className="h-8 rounded-md border border-border bg-input text-xs px-2 text-foreground"
                    >
                      <option value="all">All Status</option>
                      <option value="compliant">Compliant</option>
                      <option value="redacted">Redacted</option>
                      <option value="flagged">Flagged</option>
                    </select>
                    <Button variant="outline" size="sm" onClick={exportAuditCSV} className="h-8 text-xs" disabled={filteredAudit.length === 0}>
                      <FiDownload className="h-3 w-3 mr-1" /> Export CSV
                    </Button>
                  </div>
                </div>

                <ScrollArea className="flex-1">
                  {filteredAudit.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-center">
                      <FiShield className="h-10 w-10 text-muted-foreground/30 mb-3" />
                      <p className="text-sm text-muted-foreground">
                        {showSample ? 'No matching entries' : 'No audit entries yet. Queries will be logged here automatically.'}
                      </p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Timestamp</th>
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Query</th>
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Status</th>
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Domains</th>
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Confidence</th>
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Session</th>
                            <th className="text-left py-2 px-3 text-muted-foreground font-medium uppercase tracking-wider text-[10px]">Details</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredAudit.map((entry) => (
                            <tr key={entry.id} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                              <td className="py-2.5 px-3 whitespace-nowrap text-muted-foreground">{formatFullTimestamp(entry.timestamp)}</td>
                              <td className="py-2.5 px-3 max-w-[200px] truncate">{entry.query}</td>
                              <td className="py-2.5 px-3"><ComplianceBadge status={entry.responseStatus} /></td>
                              <td className="py-2.5 px-3">
                                <div className="flex flex-wrap gap-1">
                                  {Array.isArray(entry.domainsAccessed) && entry.domainsAccessed.map((d, i) => (
                                    <Badge key={i} variant="outline" className="text-[9px] px-1 py-0">{d}</Badge>
                                  ))}
                                </div>
                              </td>
                              <td className="py-2.5 px-3 capitalize text-muted-foreground">{entry.confidence}</td>
                              <td className="py-2.5 px-3 text-muted-foreground font-mono">{entry.sessionId.substring(0, 8)}...</td>
                              <td className="py-2.5 px-3">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 text-[10px] px-2"
                                  onClick={() => setAuditDetailEntry(entry)}
                                >
                                  <FiMaximize2 className="h-3 w-3" />
                                </Button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </ScrollArea>

                {/* Audit Detail Dialog */}
                <Dialog open={auditDetailEntry !== null} onOpenChange={() => setAuditDetailEntry(null)}>
                  <DialogContent className="bg-card border-border max-w-xl max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                      <DialogTitle className="font-serif tracking-wide">Query Detail</DialogTitle>
                    </DialogHeader>
                    {auditDetailEntry && (
                      <div className="space-y-4">
                        <div>
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Query</p>
                          <p className="text-sm bg-secondary/50 rounded-md p-3 border border-border">{auditDetailEntry.query}</p>
                        </div>
                        <div className="flex items-center gap-3">
                          <ComplianceBadge status={auditDetailEntry.responseStatus} />
                          <ConfidenceIndicator level={auditDetailEntry.confidence} />
                        </div>
                        <div>
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Response</p>
                          <div className="bg-secondary/50 rounded-md p-3 border border-border">
                            {renderMarkdown(auditDetailEntry.fullResponse)}
                          </div>
                        </div>
                        {Array.isArray(auditDetailEntry.domainsAccessed) && auditDetailEntry.domainsAccessed.length > 0 && (
                          <div>
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Domains Accessed</p>
                            <div className="flex flex-wrap gap-1.5">
                              {auditDetailEntry.domainsAccessed.map((d, i) => (
                                <Badge key={i} className="text-[10px]" style={{ backgroundColor: 'hsl(36 60% 31% / 0.2)', color: 'hsl(36 60% 50%)' }}>{d}</Badge>
                              ))}
                            </div>
                          </div>
                        )}
                        {Array.isArray(auditDetailEntry.sourcesConsulted) && auditDetailEntry.sourcesConsulted.length > 0 && (
                          <div>
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Sources Consulted</p>
                            <div className="flex flex-wrap gap-1.5">
                              {auditDetailEntry.sourcesConsulted.map((s, i) => (
                                <Badge key={i} variant="outline" className="text-[10px]">{s}</Badge>
                              ))}
                            </div>
                          </div>
                        )}
                        {Array.isArray(auditDetailEntry.flags) && auditDetailEntry.flags.length > 0 && (
                          <div>
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Flags</p>
                            <div className="flex flex-wrap gap-1.5">
                              {auditDetailEntry.flags.map((f, i) => (
                                <Badge key={i} className="text-[10px] bg-amber-900/30 text-amber-300 border-amber-700/30 hover:bg-amber-900/30">
                                  <FiAlertTriangle className="h-2.5 w-2.5 mr-1" /> {f}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                        <div className="flex items-center gap-4 text-[10px] text-muted-foreground pt-2 border-t border-border">
                          <span>Session: {auditDetailEntry.sessionId.substring(0, 12)}...</span>
                          <span>{formatFullTimestamp(auditDetailEntry.timestamp)}</span>
                        </div>
                      </div>
                    )}
                  </DialogContent>
                </Dialog>
              </div>
            )}

            {/* ==================== KNOWLEDGE BASE VIEW ==================== */}
            {activeView === 'kb' && (
              <div className="h-full flex flex-col p-4">
                <div className="mb-4">
                  <h2 className="font-serif text-lg font-semibold tracking-wide" style={{ color: 'hsl(36 60% 31%)' }}>
                    Knowledge Base Management
                  </h2>
                  <p className="text-xs text-muted-foreground mt-1">
                    Upload and manage documents for the AI sub-agents. Documents are processed and indexed for intelligent retrieval.
                  </p>
                </div>

                <ScrollArea className="flex-1">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-4xl">
                    <KBSection
                      title="Sales & Regional Knowledge Base"
                      ragId={SALES_RAG_ID}
                      docs={salesDocs}
                      setDocs={setSalesDocs}
                      loading={salesLoading}
                      setLoading={setSalesLoading}
                      statusMsg={salesStatus}
                      setStatusMsg={setSalesStatus}
                    />
                    <KBSection
                      title="HCP & Doctor Profiles Knowledge Base"
                      ragId={HCP_RAG_ID}
                      docs={hcpDocs}
                      setDocs={setHcpDocs}
                      loading={hcpLoading}
                      setLoading={setHcpLoading}
                      statusMsg={hcpStatus}
                      setStatusMsg={setHcpStatus}
                    />
                  </div>
                </ScrollArea>
              </div>
            )}
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
}
