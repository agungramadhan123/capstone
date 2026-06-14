import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Sparkles, Loader2, TrendingUp, Clock, Car, Bike } from 'lucide-react'
import { motion } from 'framer-motion'
import { useMemo } from 'react'

// Normalisasi nama kelas dari CSV (sama dengan AnalyticsDashboard)
function normalizeClass(cls: string): string {
  const map: Record<string, string> = {
    mobil: 'Mobil', motor: 'Motor', bis: 'Bis', bus: 'Bis',
    truk: 'Truk', truck: 'Truk', car: 'Mobil', motorcycle: 'Motor',
    Mobil: 'Mobil', Motor: 'Motor', Bis: 'Bis', Truk: 'Truk',
  };
  return map[cls] ?? cls;
}

export default function InsightAI() {
  const status = useAppStore(state => state.trafficStatus)
  // Ambil data CSV nyata dari Zustand store
  const analyticsData = useAppStore(state => state.analyticsData)

  // Hitung insight langsung dari data CSV
  const insights = useMemo(() => {
    if (!analyticsData || analyticsData.length === 0) return null;

    const total = analyticsData.length;
    const classCounts: Record<string, number> = { Mobil: 0, Motor: 0, Bis: 0, Truk: 0 };
    const timeline: Record<string, number> = {};

    analyticsData.forEach(row => {
      const cls = normalizeClass(row['class_name'] ?? '');
      if (classCounts[cls] !== undefined) classCounts[cls]++;

      const ts = row['timestamp'] ?? '';
      const parts = ts.split(' ');
      if (parts.length === 2) {
        const hm = parts[1].substring(0, 5);
        timeline[hm] = (timeline[hm] || 0) + 1;
      }
    });

    // Kelas terdominan
    const dominant = Object.entries(classCounts).reduce((a, b) => a[1] >= b[1] ? a : b);
    const dominantPct = total > 0 ? Math.round((dominant[1] / total) * 100) : 0;

    // Jam tersibuk
    const busiestEntry = Object.entries(timeline).reduce<[string, number] | null>(
      (max, cur) => (!max || cur[1] > max[1] ? cur : max), null
    );

    // Rasio motor ke mobil
    const motorRatio = classCounts.Mobil > 0
      ? (classCounts.Motor / classCounts.Mobil).toFixed(1)
      : '∞';

    // Kepadatan rata-rata per menit
    const minutes = Object.keys(timeline).length;
    const avgPerMin = minutes > 0 ? Math.round(total / minutes) : 0;

    return {
      total,
      classCounts,
      dominant: dominant[0],
      dominantCount: dominant[1],
      dominantPct,
      busiestTime: busiestEntry?.[0] ?? '-',
      busiestCount: busiestEntry?.[1] ?? 0,
      motorRatio,
      avgPerMin,
      minutes,
    };
  }, [analyticsData]);

  const isActive = status === 'completed' || status === 'analyzing_results';

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur h-full border-t-4 border-t-primary">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <Sparkles className="h-5 w-5 text-primary" /> AI Insights
        </CardTitle>
      </CardHeader>
      <CardContent>

        {/* Belum ada sesi analisis */}
        {status === 'idle' && (
          <div className="flex flex-col items-center justify-center h-[250px] text-muted-foreground text-center">
            <Sparkles className="h-10 w-10 mb-4 opacity-30" />
            <p className="text-sm">Insight AI akan muncul setelah sesi analisis selesai.</p>
            <p className="text-xs mt-2 opacity-60">Mulai monitoring dan tekan "Selesai & Lihat Laporan".</p>
          </div>
        )}

        {/* Sedang memproses */}
        {status === 'processing' && (
          <div className="flex flex-col items-center justify-center h-[250px] text-primary text-center">
            <Loader2 className="h-8 w-8 mb-4 animate-spin" />
            <p className="text-sm">Memproses dan membaca pola lalu lintas...</p>
          </div>
        )}

        {/* Sedang LIVE tapi belum ada data analitik */}
        {status === 'completed' && !insights && (
          <div className="flex flex-col items-center justify-center h-[250px] text-muted-foreground text-center">
            <Loader2 className="h-6 w-6 mb-3 animate-spin opacity-50" />
            <p className="text-sm">Insight akan tersedia setelah analisis selesai.</p>
          </div>
        )}

        {/* Tampilkan insight yang dihitung dari data CSV nyata */}
        {isActive && insights && (
          <div className="space-y-3">

            {/* Insight 1: Total & jam tersibuk */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="p-3.5 rounded-lg bg-primary/10 border border-primary/20"
            >
              <div className="flex items-center gap-2 mb-1 text-primary text-xs font-semibold">
                <Clock className="w-3.5 h-3.5" /> Jam Tersibuk
              </div>
              <p className="text-sm">
                Puncak lalu lintas terjadi pada pukul{' '}
                <strong className="text-white">{insights.busiestTime} WIB</strong> dengan{' '}
                <strong className="text-white">{insights.busiestCount} kendaraan</strong> per menit.
                Total sesi: <strong className="text-white">{insights.total} kendaraan</strong> dalam {insights.minutes} menit.
              </p>
            </motion.div>

            {/* Insight 2: Kelas terdominan */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25 }}
              className="p-3.5 rounded-lg bg-yellow-500/10 border border-yellow-500/20"
            >
              <div className="flex items-center gap-2 mb-1 text-yellow-400 text-xs font-semibold">
                <Bike className="w-3.5 h-3.5" /> Dominasi Kendaraan
              </div>
              <p className="text-sm">
                <strong className="text-white">{insights.dominant}</strong> mendominasi arus lalu lintas dengan{' '}
                <strong className="text-white">{insights.dominantPct}%</strong> ({insights.dominantCount} dari {insights.total} kendaraan).
              </p>
            </motion.div>

            {/* Insight 3: Rasio motor ke mobil */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="p-3.5 rounded-lg bg-green-500/10 border border-green-500/20"
            >
              <div className="flex items-center gap-2 mb-1 text-green-400 text-xs font-semibold">
                <Car className="w-3.5 h-3.5" /> Rasio Motor : Mobil
              </div>
              <p className="text-sm">
                Rasio motor terhadap mobil adalah{' '}
                <strong className="text-white">{insights.motorRatio}:1</strong>.
                Rata-rata <strong className="text-white">{insights.avgPerMin} kendaraan/menit</strong> sepanjang sesi.
              </p>
            </motion.div>

            {/* Mini breakdown tabel */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.55 }}
              className="grid grid-cols-2 gap-2"
            >
              {Object.entries(insights.classCounts).map(([cls, count]) => (
                <div key={cls} className="flex items-center justify-between bg-white/5 rounded-md px-3 py-2 text-sm">
                  <span className="text-muted-foreground">{cls}</span>
                  <span className="font-bold text-white">{count}</span>
                </div>
              ))}
            </motion.div>
          </div>
        )}

      </CardContent>
    </Card>
  )
}
