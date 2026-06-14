import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { motion } from 'framer-motion'
import { Activity } from 'lucide-react'

export default function TrafficStatus() {
  const status = useAppStore(state => state.trafficStatus)
  const condition = useAppStore(state => state.trafficCondition)

  const getStatusColor = () => {
    switch(condition) {
      case 'Lancar': return 'bg-traffic-lancar shadow-traffic-lancar/50'
      case 'Sedang': return 'bg-traffic-sedang shadow-traffic-sedang/50'
      case 'Padat': return 'bg-traffic-padat shadow-traffic-padat/50'
      case 'Sangat Padat': return 'bg-traffic-sangatPadat shadow-traffic-sangatPadat/50'
      default: return 'bg-muted'
    }
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <Activity className="h-5 w-5 text-primary"/> Status Lalu Lintas
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center py-8">
        {status === 'idle' && (
          <div className="text-muted-foreground text-center">
            <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
              <span className="text-xs">Idle</span>
            </div>
            Menunggu Analisis
          </div>
        )}
        
        {status === 'processing' && (
          <div className="text-primary text-center">
            <div className="w-24 h-24 rounded-full border-4 border-primary/20 border-t-primary animate-spin mx-auto mb-4"></div>
            Sedang Menganalisis...
          </div>
        )}

        {status === 'completed' && (
          <motion.div 
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="text-center"
          >
            <div className={`w-32 h-32 rounded-full mx-auto mb-6 flex items-center justify-center shadow-[0_0_30px_rgba(0,0,0,0.3)] animate-pulse-fast ${getStatusColor()}`}>
              <div className="w-24 h-24 rounded-full bg-background/20 backdrop-blur-sm border border-white/20"></div>
            </div>
            <h3 className="text-3xl font-bold tracking-tight uppercase text-foreground">{condition}</h3>
            <p className="text-muted-foreground mt-2">Berdasarkan volume kendaraan saat ini</p>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
}
