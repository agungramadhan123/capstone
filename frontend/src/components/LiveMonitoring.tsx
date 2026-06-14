import { useEffect } from 'react'
import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Play, Maximize2, Loader2 } from 'lucide-react'
import { Button } from './ui/button'

export default function LiveMonitoring() {
  const activeSource = useAppStore(state => state.activeSourceType)
  const camera = useAppStore(state => state.selectedCamera)
  const status = useAppStore(state => state.trafficStatus)
  const setStatus = useAppStore(state => state.setTrafficStatus)
  const vehiclesCount = useAppStore(state => state.vehiclesCount)
  const setVehiclesCount = useAppStore(state => state.setVehiclesCount)
  const setCondition = useAppStore(state => state.setTrafficCondition)

  // Start Analysis Handler
  const handleStartSim = async () => {
    setStatus('processing');
    try {
      // In a real scenario, this tells the backend to start the CV pipeline
      await fetch('http://127.0.0.1:8000/api/start', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: camera?.streamUrl || "" })
      })
      setStatus('completed');
    } catch (error) {
      console.error('Failed to start stream', error)
      setStatus('completed'); // Fallback for UI demo
    }
  }

  // Polling for live traffic status when stream is active
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (status === 'completed') {
      interval = setInterval(async () => {
        try {
          const res = await fetch('http://127.0.0.1:8000/api/traffic/status');
          if (res.ok) {
            const data = await res.json();
            setVehiclesCount(data.vehicles);
            setCondition(data.condition);
          }
        } catch (error) {
          console.error("Error fetching traffic status:", error);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [status, setVehiclesCount, setCondition]);

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur overflow-hidden group">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl">
          Live Monitoring {camera ? `- ${camera.location}` : ''}
        </CardTitle>
        <div className="flex gap-2">
          {status === 'completed' && (
            <Badge variant="outline" className="animate-pulse bg-red-500/10 text-red-500 border-red-500/20">
              ● LIVE
            </Badge>
          )}
          <Button variant="ghost" size="icon" className="h-6 w-6"><Maximize2 className="h-4 w-4"/></Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative aspect-video bg-black flex flex-col items-center justify-center border-y border-border">
          
          {status === 'idle' && (
            <>
              <Play className="h-16 w-16 text-white/50 mb-4" />
              <p className="text-white/50">
                {activeSource === 'CCTV' 
                  ? (camera ? 'Siap memulai analisis CCTV...' : 'Pilih Kamera CCTV untuk Memonitor')
                  : 'Pilih Video untuk Dianalisis'}
              </p>
              <div className="absolute bottom-4 left-4 z-20">
                <Button onClick={handleStartSim} size="sm" variant="secondary" className="backdrop-blur bg-white/10 text-white hover:bg-white/20">
                  Mulai Analisis YOLOv8
                </Button>
              </div>
            </>
          )}

          {status === 'processing' && (
            <>
              <Loader2 className="h-16 w-16 text-primary animate-spin mb-4" />
              <p className="text-white">Menghubungkan stream model...</p>
            </>
          )}

          {status === 'completed' && (
            <img 
              src="http://127.0.0.1:8000/api/video_feed" 
              alt="Live Video Stream" 
              className="w-full h-full object-cover"
              onError={(e) => {
                // Jika stream backend belum nyala, tampilkan pesan fallback
                (e.target as HTMLImageElement).style.display = 'none';
                const parent = (e.target as HTMLImageElement).parentElement;
                if (parent && !parent.querySelector('.error-msg')) {
                  const msg = document.createElement('div');
                  msg.className = 'error-msg text-red-500 text-center p-4';
                  msg.innerHTML = 'Stream gagal dimuat. Pastikan backend uvicorn berjalan.';
                  parent.appendChild(msg);
                }
              }}
            />
          )}
        </div>
        
        {/* KPI Row */}
        <div className="grid grid-cols-4 divide-x divide-border border-b border-border">
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-foreground">{vehiclesCount.mobil}</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Mobil</div>
          </div>
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-yellow-500">{vehiclesCount.motor}</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Motor</div>
          </div>
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-green-500">{vehiclesCount.bis}</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Bus</div>
          </div>
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-red-500">{vehiclesCount.truk}</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Truk</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
