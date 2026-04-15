'use client'

import Link from 'next/link'
import { useState } from 'react'
import { ArrowLeft, Download, Search } from 'lucide-react'
import axios from 'axios'

type PolicyBrowseItem = {
  document_id: string
  document_name?: string
  country?: string
  publish_date?: string
}

const countries = ['Europe', 'Australia', 'USA', 'India']
const domains = ['AI', 'Healthcare', 'Fintech', 'Crypto', 'Biotech', 'Consumer Apps', 'Insurance']

export default function ExplorePolicyPage() {
  const [browseCountry, setBrowseCountry] = useState('')
  const [browseDomain, setBrowseDomain] = useState('')
  const [browseResults, setBrowseResults] = useState<PolicyBrowseItem[]>([])
  const [isBrowsing, setIsBrowsing] = useState(false)
  const [browseError, setBrowseError] = useState('')

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:5000'

  const handleBrowse = async () => {
    if (!browseCountry && !browseDomain) {
      setBrowseError('Select at least a country or domain')
      return
    }

    setIsBrowsing(true)
    setBrowseError('')
    setBrowseResults([])

    try {
      const params: Record<string, string> = {}
      if (browseCountry) params.country = browseCountry
      if (browseDomain) params.domain = browseDomain
      const res = await axios.get<{ documents?: PolicyBrowseItem[] }>(`${apiBaseUrl}/policies`, { params })
      setBrowseResults(res.data.documents || [])
      if ((res.data.documents || []).length === 0) {
        setBrowseError('No policies found for this selection')
      }
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const message = (err.response?.data as { error?: string } | undefined)?.error
        setBrowseError(message || 'Failed to fetch policies')
      } else {
        setBrowseError('Failed to fetch policies')
      }
    } finally {
      setIsBrowsing(false)
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
    } catch {
      setBrowseError('Failed to download selected policy')
    }
  }

  const quickFilters = [
    { label: 'Europe · AI', country: 'Europe', domain: 'AI' },
    { label: 'USA · Healthcare', country: 'USA', domain: 'Healthcare' },
    { label: 'India · Fintech', country: 'India', domain: 'Fintech' },
    { label: 'Australia · Insurance', country: 'Australia', domain: 'Insurance' },
  ]

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-white">
      <div className="pointer-events-none absolute -left-24 top-10 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />
      <div className="pointer-events-none absolute right-0 top-24 h-72 w-72 rounded-full bg-sky-400/10 blur-3xl" />
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-400/10 blur-3xl" />

      <div className="relative mx-auto max-w-6xl px-6 py-10 lg:px-8">
        <header className="mb-10 rounded-[2rem] border border-white/10 bg-slate-900/90 p-8 shadow-[0_30px_120px_rgba(15,23,42,0.35)] backdrop-blur-xl">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
            <div className="max-w-3xl">
              <p className="text-sm uppercase tracking-[0.3em] text-emerald-300">Policy Search Workspace</p>
              <h1 className="mt-3 text-4xl font-semibold text-white sm:text-5xl">
                Explore policies with confidence
              </h1>
              <p className="mt-4 max-w-xl text-base leading-7 text-slate-300">
                Find the policy guidance your team needs across countries and domains, then download the full document for review.
              </p>
            </div>
            <Link
              href="/dashboard"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/10 hover:text-white"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Upload Project
            </Link>
          </div>
        </header>

        <section className="grid gap-6 xl:grid-cols-[1.4fr_0.9fr]">
          <div className="space-y-6">
            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <div className="grid gap-4 sm:grid-cols-3">
                <article className="rounded-3xl bg-slate-950/80 p-5">
                  <p className="text-sm font-medium text-slate-400">Available geographies</p>
                  <p className="mt-3 text-3xl font-semibold text-white">{countries.length}</p>
                </article>
                <article className="rounded-3xl bg-slate-950/80 p-5">
                  <p className="text-sm font-medium text-slate-400">Policy domains</p>
                  <p className="mt-3 text-3xl font-semibold text-white">{domains.length}</p>
                </article>
                <article className="rounded-3xl bg-slate-950/80 p-5">
                  <p className="text-sm font-medium text-slate-400">Instant download</p>
                  <p className="mt-3 text-3xl font-semibold text-white">Ready now</p>
                </article>
              </div>
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white">Search policies</h2>
                  <p className="mt-1 text-sm text-slate-400">Filter by country and domain to narrow down matching policy documents quickly.</p>
                </div>
                <div className="rounded-full bg-slate-950/80 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-emerald-300">
                  {browseResults.length > 0 ? `${browseResults.length} results` : 'Start browsing'}
                </div>
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <select
                  value={browseCountry}
                  onChange={(e) => setBrowseCountry(e.target.value)}
                  className="w-full rounded-3xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30"
                >
                  <option value="">All Countries</option>
                  {countries.map((country) => (
                    <option key={country} value={country}>
                      {country}
                    </option>
                  ))}
                </select>
                <select
                  value={browseDomain}
                  onChange={(e) => setBrowseDomain(e.target.value)}
                  className="w-full rounded-3xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/30"
                >
                  <option value="">All Domains</option>
                  {domains.map((domain) => (
                    <option key={domain} value={domain}>
                      {domain}
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleBrowse}
                  disabled={isBrowsing}
                  className="w-full rounded-3xl bg-blue-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Search className="h-4 w-4" />
                  {isBrowsing ? 'Searching...' : 'Browse Policies'}
                </button>
              </div>

              <div className="mt-5 space-y-3">
                <p className="text-sm font-semibold text-slate-300">Quick search examples</p>
                <div className="flex flex-wrap gap-2">
                  {quickFilters.map((filter) => (
                    <button
                      key={filter.label}
                      onClick={() => {
                        setBrowseCountry(filter.country)
                        setBrowseDomain(filter.domain)
                        setBrowseError('')
                      }}
                      className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/10 hover:text-white"
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>
              </div>

              {browseError ? (
                <p className="mt-5 rounded-3xl border border-red-500/30 bg-red-900/20 px-4 py-3 text-sm text-red-200">{browseError}</p>
              ) : (
                <p className="mt-5 text-sm text-slate-400">Browse by country or domain, then download the documents that matter most to your compliance research.</p>
              )}
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-white">Policy results</h2>
                  <p className="mt-1 text-sm text-slate-400">Organized by document name, geography, and publication date.</p>
                </div>
                {browseResults.length > 0 && (
                  <button
                    onClick={() => {
                      setBrowseCountry('')
                      setBrowseDomain('')
                      setBrowseResults([])
                      setBrowseError('')
                    }}
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-300 transition hover:bg-white/10 hover:text-white"
                  >
                    Reset search
                  </button>
                )}
              </div>

              {browseResults.length === 0 ? (
                <div className="mt-6 rounded-[1.5rem] border border-dashed border-slate-700 bg-slate-900/80 p-8 text-center">
                  <p className="text-sm font-semibold text-slate-300">Start by selecting a country, a domain, or both.</p>
                  <p className="mt-3 text-sm text-slate-400">Available policies appear here after browsing. Use the quick filters to jump-start your search.</p>
                </div>
              ) : (
                <div className="mt-6 grid gap-4">
                  {browseResults.map((doc) => (
                    <div key={doc.document_id} className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-slate-900/90 p-6 shadow-sm">
                      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                        <div className="space-y-2">
                          <p className="text-base font-semibold text-white">{doc.document_name || doc.document_id}</p>
                          <div className="flex flex-wrap gap-2 text-sm text-slate-400">
                            {doc.country && <span className="rounded-full bg-slate-950/80 px-3 py-1">{doc.country}</span>}
                            {doc.publish_date && <span className="rounded-full bg-slate-950/80 px-3 py-1">Published {doc.publish_date}</span>}
                          </div>
                        </div>
                        <button
                          onClick={() => downloadPolicy(doc.document_id, doc.document_name || doc.document_id)}
                          className="inline-flex items-center gap-2 rounded-full bg-blue-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-600"
                        >
                          <Download className="h-4 w-4" />
                          Download
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <aside className="space-y-6">
            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <h2 className="text-lg font-semibold text-white">Policy explorer tips</h2>
              <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-400">
                <li>Use a specific country filter to surface region-specific regulations.</li>
                <li>Search a domain to compare policy requirements across industries.</li>
                <li>Download the full policy text for legal review or compliance checks.</li>
              </ul>
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-slate-900/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.2)] transition hover:-translate-y-1">
              <h3 className="text-lg font-semibold text-white">How it works</h3>
              <p className="mt-3 text-sm leading-6 text-slate-400">
                Choose one or both filters, then click Browse Policies. The system returns matching policy documents that you can download for deeper analysis.
              </p>
              <div className="mt-5 grid gap-3">
                <div className="rounded-2xl bg-slate-950/80 p-4 text-sm text-slate-300">
                  <span className="block font-semibold">1.</span> Pick geography and/or domain.
                </div>
                <div className="rounded-2xl bg-slate-950/80 p-4 text-sm text-slate-300">
                  <span className="block font-semibold">2.</span> Browse available policies.
                </div>
                <div className="rounded-2xl bg-slate-950/80 p-4 text-sm text-slate-300">
                  <span className="block font-semibold">3.</span> Download the documents you need.
                </div>
              </div>
            </div>
          </aside>
        </section>
      </div>
    </main>
  )
}
