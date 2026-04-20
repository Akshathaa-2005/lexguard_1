'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Upload, AlertCircle, Shield, MessageSquare } from 'lucide-react'
import axios from 'axios'

export default function DashboardPage() {
  const router = useRouter()
  const [productDescription, setProductDescription] = useState('')
  const [selectedCountry, setSelectedCountry] = useState('')
  const [selectedDomain, setSelectedDomain] = useState('')
  const [extractedText, setExtractedText] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [legalReport, setLegalReport] = useState<any>(null)
  const [error, setError] = useState('')
  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [chatInput, setChatInput] = useState('')
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [reportTab, setReportTab] = useState<'overview' | 'policies' | 'recommendations'>('overview')
  const [expandedPolicyId, setExpandedPolicyId] = useState<string | null>(null)
  const [userEmail, setUserEmail] = useState<string | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:5000'

  const countries = ['Europe', 'Australia', 'USA', 'India']
  const domains = ['AI', 'Healthcare', 'Fintech', 'Crypto', 'Biotech', 'Consumer Apps', 'Insurance']
  const hasAnalysis = Boolean(legalReport)
  const complianceScores = legalReport?.compliance_scores || {}
  const policyPreview = legalReport?.policy_relevance || []
  const recommendations = legalReport?.recommendations || []

  useEffect(() => {
    const email = window.localStorage.getItem('lexguardUserEmail')
    if (!email) {
      router.push('/')
      return
    }
    setUserEmail(email)
  }, [router])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])


  const sendChat = async () => {
    if (!chatInput.trim() || isChatLoading) return
    const userMsg = chatInput.trim()
    setChatInput('')
    const updatedHistory: { role: 'user' | 'assistant'; content: string }[] = [...chatMessages, { role: 'user', content: userMsg }]
    setChatMessages(updatedHistory)
    setIsChatLoading(true)

    try {
      const res = await axios.post(`${apiBaseUrl}/chat`, {
        message: userMsg,
        context: legalReport || {},
        product_description: productDescription,
        history: updatedHistory.slice(-10),
      })
      setChatMessages(prev => [...prev, { role: 'assistant', content: res.data.reply }])
    } catch {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Please upload or enter your product description first so I can assist you better.' }])
    } finally {
      setIsChatLoading(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(`${apiBaseUrl}/upload`, formData)
      setExtractedText(response.data.extracted_text)
      setProductDescription(response.data.extracted_text)
    } catch (err) {
      setError('We could not read your file. Please upload a valid PDF or DOCX document and try again.')
      console.error(err)
    }
  }

  const handleAnalyze = async () => {
    if (!productDescription.trim()) {
      setError('Please enter your product or project description before starting the analysis.')
      return
    }

    setIsAnalyzing(true)
    setError('')

    try {
      const response = await axios.post(`${apiBaseUrl}/analyze`, {
        product_description: productDescription,
        country: selectedCountry,
        domain: selectedDomain,
      })

      setLegalReport(response.data)
      setChatMessages([
        {
          role: 'assistant',
          content: `I've completed the legal analysis. Validity score: ${response.data.validity_score}/100, Risk: ${response.data.risk_level}.

${response.data.executive_summary}

Ask me anything about legal requirements or how to address risks.`,
        },
      ])
    } catch (err: any) {
      setError(err.response?.data?.error || 'We were unable to complete the legal analysis right now. Please try again in a few moments.')
      console.error(err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const downloadPolicy = async (documentId: string, policyName: string) => {
    try {
      const response = await axios.get(`${apiBaseUrl}/policy/${documentId}`, {
        responseType: 'blob',
      })

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `policy_${policyName.replace(/\s+/g, '_')}.txt`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError('Failed to download policy document')
      console.error(err)
    }
  }

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'Low':
        return 'text-green-400'
      case 'Medium':
        return 'text-yellow-400'
      case 'High':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  const signOut = () => {
    window.localStorage.removeItem('lexguardUserEmail')
    router.push('/')
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-white">
      <div className="pointer-events-none absolute -left-24 top-10 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />
      <div className="pointer-events-none absolute right-0 top-24 h-72 w-72 rounded-full bg-sky-400/10 blur-3xl" />
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-400/10 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 py-10 lg:px-8">
        <header className="mb-10 rounded-[2rem] border border-white/10 bg-slate-900/90 p-8 shadow-[0_30px_120px_rgba(15,23,42,0.35)] backdrop-blur-xl">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-3">
              <p className="text-sm uppercase tracking-[0.3em] text-emerald-300">Welcome back</p>
              <h1 className="text-4xl font-semibold text-white sm:text-5xl">Your LexGuard workspace</h1>
              <p className="max-w-3xl text-base leading-7 text-slate-300">Signed in with Gmail for a consistent, secure legal compliance workflow.</p>
            </div>
            <div className="flex flex-col gap-3">
              <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5 text-sm text-slate-300 shadow-inner shadow-slate-950/10">
                <p className="font-semibold text-white">Signed in as</p>
                <p className="mt-1 text-sm text-slate-300">{userEmail || 'loading...'}</p>
              </div>
              <button
                onClick={signOut}
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-slate-300 transition hover:bg-white/10 hover:text-white"
              >
                Sign out
              </button>
            </div>
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              { label: 'Current score', value: legalReport?.validity_score ? `${legalReport.validity_score}/100` : 'Awaiting analysis' },
              { label: 'Policy matches', value: legalReport ? `${legalReport.policy_relevance.length} found` : 'No search yet' },
              { label: 'Risk status', value: legalReport ? legalReport.risk_level : 'Not evaluated' },
            ].map((item) => (
              <div key={item.label} className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-6 transition hover:-translate-y-1">
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">{item.label}</p>
                <p className="mt-3 text-2xl font-semibold text-white">{item.value}</p>
              </div>
            ))}
          </div>
        </header>

        <div className="grid gap-6 xl:grid-cols-[0.45fr_0.55fr]">
          <div className="space-y-6">
            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3 text-emerald-300">
                  <MessageSquare className="h-5 w-5" />
                  <p className="text-sm uppercase tracking-[0.3em]">Legal chat</p>
                </div>
                <button
                  onClick={() => setChatMessages([])}
                  className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-slate-300 transition hover:border-slate-200 hover:text-white"
                >
                  Clear history
                </button>
              </div>
              <div className="mt-6 flex h-[970px] flex-col rounded-[1.75rem] border border-slate-800 bg-slate-950/95 p-4 shadow-inner shadow-slate-950/20">
                <div className="flex-1 space-y-4 overflow-y-auto pr-2 text-sm">
                  {chatMessages.length === 0 ? (
                    <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/80 p-6 text-slate-400">
                      <p className="font-semibold text-white">Your chat assistant is ready.</p>
                      <p className="mt-2">Ask questions about your project idea, analysis results, and policy guidance.</p>
                    </div>
                  ) : (
                    chatMessages.map((message, index) => (
                      <div
                        key={index}
                        className={`rounded-[1.5rem] px-4 py-3 ${message.role === 'user' ? 'bg-slate-800 text-slate-100 self-end' : 'bg-slate-700/80 text-slate-200 self-start'} max-w-[90%]`}
                      >
                        <p className="text-[0.72rem] uppercase tracking-[0.26em] text-slate-400">
                          {message.role === 'user' ? 'You' : 'LexGuard'}
                        </p>
                        <p className="mt-2 whitespace-pre-wrap leading-6">{message.content}</p>
                      </div>
                    ))
                  )}
                  <div ref={chatEndRef} />
                </div>
                <div className="mt-4 border-t border-slate-800 pt-4">
                  <label htmlFor="dashboard-chat" className="sr-only">Type a message</label>
                  <textarea
                    id="dashboard-chat"
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    rows={4}
                    className="min-h-[100px] w-full rounded-3xl border border-slate-800 bg-slate-950 px-4 py-3 text-sm text-white outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30"
                    placeholder="Ask LexGuard about your idea, risks, or policies..."
                  />
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <p className="text-xs text-slate-400">Your latest project inputs are included in the chat context.</p>
                    <button
                      onClick={sendChat}
                      disabled={isChatLoading || !chatInput.trim()}
                      className="rounded-full bg-blue-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {isChatLoading ? 'Sending…' : 'Send'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <div className="flex items-center gap-3 text-sky-300">
                <Shield className="h-5 w-5" />
                <p className="text-sm uppercase tracking-[0.3em]">Explore policies</p>
              </div>
              <h2 className="mt-5 text-xl font-semibold text-white">Policy exploration</h2>
              <p className="mt-3 text-sm text-slate-400">Go to the dedicated policy explorer page for filtered searches and policy guidance.</p>
              <div className="mt-6 rounded-[1.75rem] border border-slate-800 bg-slate-950/80 p-5">
                <p className="text-sm text-slate-300">Browse by country, domain, and policy type with a search-first experience built for legal risk discovery.</p>
                <ul className="mt-4 space-y-3 text-sm text-slate-400">
                  <li>• Quick search across jurisdictional policies</li>
                  <li>• Topic-based filtering for your industry</li>
                  <li>• Download policy references instantly</li>
                </ul>
              </div>
              <button
                onClick={() => router.push('/explore-policy')}
                className="mt-6 w-full rounded-3xl bg-blue-500 px-5 py-3 font-semibold text-white transition hover:bg-blue-600"
              >
                Open policy explorer
              </button>
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <div className="flex items-center gap-3 text-emerald-300">
                <Upload className="h-5 w-5" />
                <p className="text-sm uppercase tracking-[0.3em]">Upload project idea</p>
              </div>
              <h2 className="mt-5 text-xl font-semibold text-white">Start legal analysis</h2>
              <p className="mt-3 text-sm text-slate-400">Upload a brief or paste your idea and receive tailored legal insight.</p>
              <div className="mt-5 space-y-5">
                <label className="block text-sm font-medium text-slate-300">Product description</label>
                <textarea
                  value={productDescription}
                  onChange={e => setProductDescription(e.target.value)}
                  className="w-full min-h-[140px] rounded-3xl border border-slate-700 bg-slate-950 px-4 py-4 text-white outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30"
                  placeholder="Describe your product idea..."
                />
                <label className="block text-sm font-medium text-slate-300">Document upload</label>
                <input
                  type="file"
                  accept=".pdf,.docx"
                  onChange={handleFileUpload}
                  className="w-full rounded-3xl border border-slate-700 bg-slate-950 px-4 py-4 text-white file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-400"
                />
                <div className="grid gap-4 sm:grid-cols-2">
                  <select
                    value={selectedCountry}
                    onChange={e => setSelectedCountry(e.target.value)}
                    className="w-full rounded-3xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30"
                  >
                    <option value="">Select country</option>
                    {countries.map(country => <option key={country} value={country}>{country}</option>)}
                  </select>
                  <select
                    value={selectedDomain}
                    onChange={e => setSelectedDomain(e.target.value)}
                    className="w-full rounded-3xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30"
                  >
                    <option value="">Select domain</option>
                    {domains.map(domain => <option key={domain} value={domain}>{domain}</option>)}
                  </select>
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full rounded-3xl bg-blue-500 px-5 py-3 font-semibold text-white transition hover:bg-blue-600 disabled:opacity-50"
                >
                  {isAnalyzing ? 'Analyzing…' : 'Run analysis'}
                </button>
                {error && (
                  <div className="rounded-3xl border border-red-500/30 bg-red-900/20 p-4 text-sm text-red-200">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-red-300" />
                      <span>{error}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {hasAnalysis ? (
          <section className="mt-10 space-y-6">
            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)]">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-emerald-300">Analysis dashboard</p>
                  <h2 className="mt-3 text-2xl font-semibold text-white">Insight hubs</h2>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Explore the compliance results, policy briefing, and recommendations generated from your latest analysis.</p>
                </div>
                <div className="flex flex-wrap gap-2 rounded-full bg-slate-950/80 p-1">
                  {[
                    { id: 'overview', label: 'Overview' },
                    { id: 'policies', label: 'Policies' },
                    { id: 'recommendations', label: 'Recommendations' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setReportTab(tab.id as 'overview' | 'policies' | 'recommendations')}
                      className={`rounded-full px-4 py-2 text-sm font-semibold transition ${reportTab === tab.id ? 'bg-blue-500 text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white'}`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {[
                  { label: 'Regulatory score', value: `${legalReport.validity_score ?? 'N/A'}/100`, accent: 'bg-emerald-500/10 text-emerald-300' },
                  { label: 'Risk status', value: legalReport.risk_level || 'Unknown', accent: getRiskLevelColor(legalReport.risk_level) },
                  { label: 'Policy matches', value: `${policyPreview.length} found`, accent: 'bg-sky-500/10 text-sky-300' },
                  { label: 'Recommendations', value: `${recommendations.length} items`, accent: 'bg-violet-500/10 text-violet-300' },
                ].map((metric) => (
                  <div key={metric.label} className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-400">{metric.label}</p>
                    <p className={`mt-4 text-2xl font-semibold ${metric.accent}`}>{metric.value}</p>
                  </div>
                ))}
              </div>

              <div className="mt-6 rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-6">
                {reportTab === 'overview' && (
                  <div className="grid gap-6 xl:grid-cols-[0.6fr_0.4fr]">
                    <div className="space-y-5">
                      <div className="rounded-[1.75rem] border border-white/10 bg-slate-900/90 p-5">
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Executive summary</p>
                        <p className="mt-3 leading-7 text-slate-300">{legalReport.executive_summary || 'No executive summary is available yet.'}</p>
                      </div>
                      <div className="grid gap-4 sm:grid-cols-2">
                        {[
                          { label: 'Data privacy', value: complianceScores.data_privacy ?? 'N/A' },
                          { label: 'AI regulation', value: complianceScores.ai_regulation ?? 'N/A' },
                          { label: 'Financial regulation', value: complianceScores.financial_regulation ?? 'N/A' },
                          { label: 'Consumer protection', value: complianceScores.consumer_protection ?? 'N/A' },
                        ].map((item) => (
                          <div key={item.label} className="rounded-[1.75rem] border border-white/10 bg-slate-900/90 p-5">
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-400">{item.label}</p>
                            <p className="mt-4 text-3xl font-semibold text-white">{typeof item.value === 'number' ? `${item.value}/100` : item.value}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-5">
                      <div className="rounded-[1.75rem] border border-white/10 bg-slate-900/90 p-5">
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Risk overview</p>
                        <p className={`mt-4 text-4xl font-semibold ${getRiskLevelColor(legalReport.risk_level)}`}>{legalReport.risk_level || 'Unknown'}</p>
                        <p className="mt-3 text-sm leading-6 text-slate-300">This risk assessment reflects policy sensitivity and compliance exposure for your idea.</p>
                      </div>
                      <div className="rounded-[1.75rem] border border-white/10 bg-slate-900/90 p-5">
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Action readiness</p>
                        <p className="mt-4 text-3xl font-semibold text-white">{recommendations.length > 0 ? `${recommendations.length} steps` : 'No actions'}</p>
                        <p className="mt-3 text-sm leading-6 text-slate-300">Use the recommendations tab to review the suggested next steps and compliance actions.</p>
                      </div>
                    </div>
                  </div>
                )}

                {reportTab === 'policies' && (
                  <div className="space-y-4">
                    <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5 text-slate-300">
                      <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Policy briefing</p>
                      <p className="mt-3 text-sm leading-6">Review the top policy matches below, expand details, and download the relevant regulatory text.</p>
                    </div>
                    <div className="space-y-4">
                      {policyPreview.length > 0 ? policyPreview.slice(0, 5).map((policy: any) => (
                        <div key={policy.document_id} className="rounded-[1.75rem] border border-white/10 bg-slate-900/90 p-5">
                          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                            <div className="space-y-2">
                              <p className="text-sm uppercase tracking-[0.3em] text-slate-400">{policy.country || 'Policy'}</p>
                              <p className="text-lg font-semibold text-white">{policy.policy_name || policy.document_id}</p>
                              <p className="text-sm text-slate-400">Relevance {policy.relevance_score?.toFixed?.(1) ?? policy.relevance_score ?? 'N/A'}%</p>
                            </div>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => setExpandedPolicyId(expandedPolicyId === policy.document_id ? null : policy.document_id)}
                                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/10"
                              >
                                {expandedPolicyId === policy.document_id ? 'Hide details' : 'Show details'}
                              </button>
                              <button
                                onClick={() => downloadPolicy(policy.document_id, policy.policy_name || policy.document_id)}
                                className="rounded-full bg-blue-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-600"
                              >
                                Download
                              </button>
                            </div>
                          </div>
                          {expandedPolicyId === policy.document_id && policy.summary && (
                            <p className="mt-4 text-sm leading-6 text-slate-300">{policy.summary}</p>
                          )}
                        </div>
                      )) : (
                        <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5 text-slate-400">No retrieved policies are available yet. Run an analysis to preview relevant laws.</div>
                      )}
                    </div>
                  </div>
                )}

                {reportTab === 'recommendations' && (
                  <div className="space-y-4">
                    <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5 text-slate-300">
                      <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Recommendations</p>
                      <p className="mt-3 text-sm leading-6">Focus on the highest-impact actions for compliance and risk mitigation.</p>
                    </div>
                    <div className="space-y-4">
                      {recommendations.length > 0 ? recommendations.map((rec: string, index: number) => (
                        <div key={index} className="rounded-[1.75rem] border border-white/10 bg-slate-900/90 p-5">
                          <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Step {index + 1}</p>
                          <p className="mt-3 text-sm leading-7 text-slate-300">{rec}</p>
                        </div>
                      )) : (
                        <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5 text-slate-400">No recommendations were generated for this analysis.</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        ) : (
          <section className="mt-10 rounded-[2rem] border border-dashed border-white/10 bg-slate-900/80 p-8 text-center text-slate-400">
            Run an analysis to unlock your compliance report, risk insights, policy preview, and recommendations.
          </section>
        )}
      </div>
    </main>
  )
}
