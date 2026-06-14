import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts'
import { BarChart2 } from 'lucide-react'

const mockData = [
  { time: '06:00', volume: 400 },
  { time: '08:00', volume: 1200 },
  { time: '10:00', volume: 800 },
  { time: '12:00', volume: 950 },
  { time: '14:00', volume: 850 },
  { time: '16:00', volume: 1400 },
  { time: '18:00', volume: 1842 }, // Peak
  { time: '20:00', volume: 1100 },
  { time: '22:00', volume: 500 },
]

export default function TrafficAnalytics() {
  const status = useAppStore(state => state.trafficStatus)

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur w-full h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <BarChart2 className="h-5 w-5 text-primary" /> Traffic Analytics
        </CardTitle>
      </CardHeader>
      <CardContent>
        {status === 'idle' ? (
          <div className="h-[350px] w-full flex flex-col items-center justify-center text-muted-foreground bg-muted/20 rounded-md border border-dashed border-border">
            <BarChart2 className="h-12 w-12 mb-4 opacity-50" />
            <p>Data analitik belum tersedia.</p>
            <p className="text-sm">Silakan mulai Live Monitoring untuk melihat grafik.</p>
          </div>
        ) : (
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
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
                <ReferenceDot 
                  x="18:00" 
                  y={1842} 
                  r={6} 
                  fill="red" 
                  stroke="none" 
                  label={{ position: 'top', value: 'Peak Traffic', fill: 'hsl(var(--foreground))', fontSize: 12 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
