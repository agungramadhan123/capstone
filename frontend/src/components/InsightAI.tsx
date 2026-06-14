import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Sparkles, Loader2 } from 'lucide-react'
import { motion } from 'framer-motion'

export default function InsightAI() {
  const status = useAppStore(state => state.trafficStatus)

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur h-full border-t-4 border-t-primary">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <Sparkles className="h-5 w-5 text-primary" /> AI Insights
        </CardTitle>
      </CardHeader>
      <CardContent>
        {status === 'idle' ? (
          <div className="flex flex-col items-center justify-center h-[250px] text-muted-foreground text-center">
            <Sparkles className="h-10 w-10 mb-4 opacity-30" />
            <p className="text-sm">Insight AI akan muncul setelah data analisis terkumpul secara otomatis.</p>
          </div>
        ) : status === 'processing' ? (
          <div className="flex flex-col items-center justify-center h-[250px] text-primary text-center">
            <Loader2 className="h-8 w-8 mb-4 animate-spin" />
            <p className="text-sm">Memproses dan membaca pola lalu lintas...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="p-4 rounded-lg bg-primary/10 border border-primary/20"
            >
              <p className="text-sm">Jam tersibuk terjadi pada pukul <strong>18.00 WIB</strong> dengan volume <strong>1842 kendaraan</strong>.</p>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20"
            >
              <p className="text-sm">Volume kendaraan meningkat <strong>32%</strong> dibanding rata-rata harian minggu lalu.</p>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
              className="p-4 rounded-lg bg-green-500/10 border border-green-500/20"
            >
              <p className="text-sm">Rasio kendaraan roda dua mendominasi sebesar <strong>71%</strong> dari total arus lalu lintas.</p>
            </motion.div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
