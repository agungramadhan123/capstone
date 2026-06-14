import { useAppStore } from '../store'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, PieChart, Pie, Cell, LineChart, Line, Legend } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Car, Bike, Bus, Truck, TrendingUp, Clock, ArrowRight } from 'lucide-react'

const COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444']; // Biru, Kuning, Hijau, Merah
const CLASS_LABELS = ['Mobil', 'Motor', 'Bis', 'Truk'];

// Normalisasi nama kelas kendaraan dari CSV (kadang beda casing/spelling)
function normalizeClass(cls: string): string {
  const map: Record<string, string> = {
    mobil: 'Mobil', motor: 'Motor', bis: 'Bis', bus: 'Bis',
    truk: 'Truk', truck: 'Truk', car: 'Mobil', motorcycle: 'Motor',
    Mobil: 'Mobil', Motor: 'Motor', Bis: 'Bis', Truk: 'Truk',
  };
  return map[cls] ?? cls;
}

export default function AnalyticsDashboard() {
  const data = useAppStore(state => state.analyticsData);

  const { barData, pieData, timelineData, summary } = useMemo(() => {
    if (!data || data.length === 0) return { barData: [], pieData: [], timelineData: [], summary: null };

    const classCounts: Record<string, number> = { Mobil: 0, Motor: 0, Bis: 0, Truk: 0 };
    const timeline: Record<string, Record<string, any>> = {};
    const directionCounts: Record<string, number> = {};
    let firstTime = '';
    let lastTime = '';

    data.forEach(row => {
      // Normalisasi: CSV memakai kolom 'class_name' (lowercase key dari DictReader)
      const rawCls = row['class_name'] ?? row['Class_Name'] ?? row['class'] ?? '';
      const cls = normalizeClass(rawCls);

      if (classCounts[cls] !== undefined) {
        classCounts[cls]++;
      } else {
        classCounts[cls] = 1;
      }

      // Hitung distribusi arah
      const dir = row['direction'] ?? '';
      if (dir) directionCounts[dir] = (directionCounts[dir] || 0) + 1;

      // Timestamp format dari CSV: "YYYY-MM-DD HH:MM:SS"
      const ts = row['timestamp'] ?? '';
      const timeParts = ts.split(' ');
      if (timeParts.length === 2) {
        const hm = timeParts[1].substring(0, 5); // HH:MM
        if (!firstTime || hm < firstTime) firstTime = hm;
        if (!lastTime || hm > lastTime) lastTime = hm;

        if (!timeline[hm]) {
          timeline[hm] = { time: hm, Mobil: 0, Motor: 0, Bis: 0, Truk: 0, Total: 0 };
        }
        if (timeline[hm][cls] !== undefined) {
          timeline[hm][cls]++;
        }
        timeline[hm].Total++;
      }
    });

    const barData = CLASS_LABELS.map(key => ({
      name: key,
      Total: classCounts[key] ?? 0,
    }));

    const pieData = barData.filter(d => d.Total > 0);

    const timelineData = Object.values(timeline).sort((a: any, b: any) =>
      a.time.localeCompare(b.time)
    );

    // Cari jam tersibuk
    const busiestEntry = timelineData.reduce<any>((max, cur: any) =>
      cur.Total > (max?.Total ?? 0) ? cur : max, null);

    // Kelas terdominan
    const dominantClass = CLASS_LABELS.reduce((a, b) =>
      classCounts[a] >= classCounts[b] ? a : b);
    const dominantPct = data.length > 0
      ? Math.round((classCounts[dominantClass] / data.length) * 100)
      : 0;

    const summary = {
      total: data.length,
      classCounts,
      busiestTime: busiestEntry?.time ?? '-',
      busiestCount: busiestEntry?.Total ?? 0,
      dominantClass,
      dominantPct,
      firstTime,
      lastTime,
    };

    return { barData, pieData, timelineData, summary };
  }, [data]);

  if (!data) return (
    <div className="p-12 flex flex-col items-center justify-center h-full text-muted-foreground">
      <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full mb-4" />
      <p>Memuat data laporan dari backend...</p>
    </div>
  );

  if (data.length === 0) return (
    <div className="p-12 flex flex-col items-center justify-center h-full text-muted-foreground text-center">
      <TrendingUp className="h-12 w-12 mb-4 opacity-30" />
      <p className="font-semibold">Tidak ada data kendaraan yang terekam pada sesi ini.</p>
      <p className="text-sm mt-1 text-muted-foreground/60">Pastikan sistem berhasil mendeteksi kendaraan yang melewati zona deteksi.</p>
    </div>
  );

  return (
    <div className="flex flex-col gap-4 p-4 w-full h-full bg-black/40 overflow-y-auto">

      {/* ─── KPI Summary Cards dari data CSV nyata ─── */}
      {summary && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-3"
        >
          {[
            { label: 'Total Kendaraan', value: summary.total, sub: `${summary.firstTime} – ${summary.lastTime}`, icon: <TrendingUp className="w-4 h-4" />, color: 'text-primary' },
            { label: 'Mobil', value: summary.classCounts.Mobil, sub: `${Math.round((summary.classCounts.Mobil / summary.total) * 100)}% dari total`, icon: <Car className="w-4 h-4" />, color: 'text-blue-400' },
            { label: 'Motor', value: summary.classCounts.Motor, sub: `${Math.round((summary.classCounts.Motor / summary.total) * 100)}% dari total`, icon: <Bike className="w-4 h-4" />, color: 'text-yellow-400' },
            { label: 'Jam Tersibuk', value: summary.busiestTime, sub: `${summary.busiestCount} kendaraan/menit`, icon: <Clock className="w-4 h-4" />, color: 'text-green-400' },
          ].map((kpi, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.08 }}
              className="bg-white/5 border border-white/10 rounded-lg p-3 flex flex-col gap-1"
            >
              <div className={`flex items-center gap-1.5 text-xs font-medium ${kpi.color} opacity-80`}>
                {kpi.icon} {kpi.label}
              </div>
              <div className="text-2xl font-bold text-white">{kpi.value}</div>
              <div className="text-xs text-white/40">{kpi.sub}</div>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* ─── Baris Grafik: Bar Chart + Pie Chart ─── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="bg-card/50 border-border/50 backdrop-blur">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-white">Total per Kelas Kendaraan</CardTitle>
          </CardHeader>
          <CardContent className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" vertical={false} />
                <XAxis dataKey="name" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#444', borderRadius: '8px', color: '#fff' }}
                  cursor={{ fill: '#333' }}
                  formatter={(value: any) => [`${value} kendaraan`, 'Total']}
                />
                <Bar dataKey="Total" radius={[4, 4, 0, 0]}>
                  {barData.map((_, index) => (
                    <Cell key={index} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50 backdrop-blur">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-white">Komposisi Lalu Lintas</CardTitle>
          </CardHeader>
          <CardContent className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="Total"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={35}
                  outerRadius={75}
                  paddingAngle={4}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {pieData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#444', borderRadius: '8px', color: '#fff' }}
                  formatter={(value: any, name: any) => [`${value} kendaraan`, name]}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#aaa' }} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* ─── Line Chart: Tren Kepadatan Per Menit ─── */}
      <Card className="bg-card/50 border-border/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-white">
            Tren Kepadatan Per Menit
            {summary && (
              <span className="ml-2 text-xs font-normal text-white/40">
                {summary.firstTime} – {summary.lastTime} · {summary.total} kendaraan total
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timelineData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
              <XAxis dataKey="time" stroke="#666" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#666" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#444', borderRadius: '8px', color: '#fff' }}
                labelStyle={{ color: '#aaa', fontWeight: 'bold' }}
                formatter={(value: any, name: any) => [`${value}`, name]}
              />
              <Legend iconType="plainline" wrapperStyle={{ fontSize: '11px', color: '#aaa' }} />
              <Line type="monotone" dataKey="Total" stroke="#10b981" strokeWidth={3} dot={{ r: 3, fill: '#10b981', strokeWidth: 0 }} name="Total" />
              <Line type="monotone" dataKey="Mobil" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Mobil" />
              <Line type="monotone" dataKey="Motor" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Motor" />
              <Line type="monotone" dataKey="Bis" stroke="#10b981" strokeWidth={1.5} dot={false} name="Bus" strokeDasharray="4 2" />
              <Line type="monotone" dataKey="Truk" stroke="#ef4444" strokeWidth={1.5} dot={false} name="Truk" strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* ─── Tabel Raw Data (5 baris pertama) ─── */}
      {data.length > 0 && (
        <Card className="bg-card/50 border-border/50 backdrop-blur">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-white flex items-center justify-between">
              <span>Sample Data CSV ({data.length} baris total)</span>
              <span className="text-xs font-normal text-white/40">5 event pertama</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-white/70">
                <thead>
                  <tr className="border-b border-white/10 text-white/40">
                    <th className="text-left py-1.5 pr-3">Timestamp</th>
                    <th className="text-left py-1.5 pr-3">ID</th>
                    <th className="text-left py-1.5 pr-3">Kelas</th>
                    <th className="text-left py-1.5 pr-3">Conf</th>
                    <th className="text-left py-1.5">Arah</th>
                  </tr>
                </thead>
                <tbody>
                  {data.slice(0, 5).map((row, i) => (
                    <tr key={i} className="border-b border-white/5">
                      <td className="py-1.5 pr-3 font-mono">{row.timestamp}</td>
                      <td className="py-1.5 pr-3">{row.vehicle_id}</td>
                      <td className="py-1.5 pr-3 font-medium text-white">{normalizeClass(row.class_name)}</td>
                      <td className="py-1.5 pr-3">{parseFloat(row.confidence).toFixed(2)}</td>
                      <td className="py-1.5 text-white/50">{row.direction}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
