import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useAppStore } from './store'
import { LayoutDashboard, Video, Map, Activity, MessageSquare } from 'lucide-react'

// Placeholder for components that we will build
import HeroSection from './components/HeroSection'
import InputSource from './components/InputSource'
import LiveMonitoring from './components/LiveMonitoring'
import TrafficStatus from './components/TrafficStatus'
import MapMonitoring from './components/MapMonitoring'
import TrafficAnalytics from './components/TrafficAnalytics'
import InsightAI from './components/InsightAI'
import TrafficAssistant from './components/TrafficAssistant'
import UploadHistorySection from './components/UploadHistorySection'

function App() {
  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <nav className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 flex h-14 items-center justify-between">
          <div className="flex items-center space-x-2 cursor-pointer" onClick={() => scrollTo('hero')}>
            <Activity className="h-6 w-6 text-primary" />
            <span className="font-bold text-lg tracking-tight">Smart Traffic</span>
          </div>
          <div className="hidden md:flex items-center space-x-6 text-sm font-medium text-muted-foreground">
            <button onClick={() => scrollTo('monitor')} className="hover:text-foreground transition-colors flex items-center gap-2"><Video size={16}/> Monitor</button>
            <button onClick={() => scrollTo('map')} className="hover:text-foreground transition-colors flex items-center gap-2"><Map size={16}/> Map</button>
            <button onClick={() => scrollTo('analytics')} className="hover:text-foreground transition-colors flex items-center gap-2"><LayoutDashboard size={16}/> Analytics</button>
            <button onClick={() => scrollTo('assistant')} className="hover:text-foreground transition-colors flex items-center gap-2"><MessageSquare size={16}/> AI Assistant</button>
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-8 space-y-24">
        {/* Section 1: Hero */}
        <section id="hero">
          <HeroSection />
        </section>

        {/* Section 2: Input Source */}
        <section id="input">
          <InputSource />
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8" id="monitor">
          <div className="lg:col-span-2 space-y-8">
            {/* Section 3: Live Monitoring */}
            <LiveMonitoring />
          </div>
          <div className="space-y-8">
            {/* Section 4: Traffic Status */}
            <TrafficStatus />
          </div>
        </div>

        {/* Section 5: Map Monitoring */}
        <section id="map">
          <MapMonitoring />
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8" id="analytics">
          <div className="lg:col-span-2">
            {/* Section 6: Analytics */}
            <TrafficAnalytics />
          </div>
          <div className="space-y-8">
            {/* Section 7: Insight AI */}
            <InsightAI />
          </div>
        </div>

        {/* Section 8: Traffic Assistant */}
        <section id="assistant">
          <TrafficAssistant />
        </section>

        {/* Section 9: Upload History */}
        <section id="history">
          <UploadHistorySection />
        </section>

      </main>

      <footer className="border-t py-6 mt-24">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          &copy; {new Date().getFullYear()} Smart Traffic Bandung. All rights reserved.
        </div>
      </footer>
    </div>
  )
}

export default App
