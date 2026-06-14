import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { BarChart2, Loader2 } from 'lucide-react'
import { useEffect, useState, useMemo } from 'react'

export default function TrafficAnalytics() {
  const status = useAppStore(state => state.trafficStatus)
  const [masterData, setMasterData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  // Ambil data log master saat: halaman dimuat pertama kali (idle) ATAU setelah sesi selesai (analyzing_results)
  useEffect(() => {
    const shouldFetch = status === 'idle' || status === 'analyzing_results';
    if (!shouldFetch) return;

    setLoading(true)
    fetch('http://127.0.0.1:8000/api/logs/master')
      .then(res => res.json())
      .then(res => {
        if (res.status === 'success' && Array.isArray(res.data)) {
          setMasterData(res.data)
        }
      })
      .catch(err => console.error("Gagal mengambil master data:", err))
      .finally(() => setLoading(false))
  }, [status])

  // Hitung volume per menit
  const chartData = useMemo(() => {
    if (!masterData || masterData.length === 0) return [];
    const timeline: Record<string, number> = {};
    masterData.forEach(row => {
      const timeParts = row.timestamp?.split(' ');
      if (timeParts && timeParts.length === 2) {
        const hm = timeParts[1].substring(0, 5); // Ambil jam:menit (HH:MM)
        timeline[hm] = (timeline[hm] || 0) + 1;
      }
    });
    return Object.keys(timeline).sort().map(time => ({ time, volume: timeline[time] }));
  }, [masterData]);

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur w-full h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <BarChart2 className="h-5 w-5 text-primary" /> Traffic Analytics (Semua Sesi)
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="h-[350px] w-full flex flex-col items-center justify-center text-muted-foreground bg-muted/20 rounded-md border border-dashed border-border">
             <Loader2 className="h-8 w-8 animate-spin mb-4 opacity-50" />
             <p>Memuat data riwayat lalu lintas...</p>
          </div>
        ) : chartData.length === 0 ? (
          <div className="h-[350px] w-full flex flex-col items-center justify-center text-muted-foreground bg-muted/20 rounded-md border border-dashed border-border">
            <BarChart2 className="h-12 w-12 mb-4 opacity-50" />
            <p>Data analitik belum tersedia.</p>
            <p className="text-sm">Selesaikan minimal satu sesi Live Monitoring untuk menyimpan riwayat.</p>
          </div>
        ) : (
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
                  itemStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="volume" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={3}
                  dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2 }}
                  activeDot={{ r: 8 }}
                  animationDuration={2000}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
