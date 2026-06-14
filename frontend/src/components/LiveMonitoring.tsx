import { useEffect, useRef } from 'react'
import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Play, Maximize2, Loader2, XCircle, AlertCircle, RefreshCw } from 'lucide-react'
import { Button } from './ui/button'
import AnalyticsDashboard from './AnalyticsDashboard'

export default function LiveMonitoring() {
  const activeSource = useAppStore(state => state.activeSourceType)
  const activeCctvUrl = useAppStore(state => state.activeCctvUrl)
  const uploadedFileUrl = useAppStore(state => state.uploadedFileUrl)
  const status = useAppStore(state => state.trafficStatus)
  const setStatus = useAppStore(state => state.setTrafficStatus)
  const vehiclesCount = useAppStore(state => state.vehiclesCount)
  const setVehiclesCount = useAppStore(state => state.setVehiclesCount)
  const setCondition = useAppStore(state => state.setTrafficCondition)
  const setAnalyticsData = useAppStore(state => state.setAnalyticsData)

  // FIX Bug #5: isReady harus strict — CCTV butuh URL, Upload butuh path file
  const isReady = activeSource === 'CCTV'
    ? (!!activeCctvUrl && activeCctvUrl.trim() !== '')
    : (!!uploadedFileUrl && uploadedFileUrl.trim() !== '');

  const containerRef = useRef<HTMLDivElement>(null);

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  // FIX Bug #2 & #5: Validasi kuat + tidak berpura-pura sukses saat error
  const handleStartSim = async () => {
    // Guard: pastikan ada source sebelum memanggil API
    const targetUrl = activeSource === 'CCTV' ? activeCctvUrl : uploadedFileUrl;
    if (!targetUrl || targetUrl.trim() === '') {
      alert(
        activeSource === 'CCTV'
          ? 'Harap masukkan URL CCTV terlebih dahulu di bagian "Pilih Sumber Video".'
          : 'Harap unggah file video terlebih dahulu sebelum memulai analisis.'
      );
      return;
    }

    setStatus('processing');
    try {
      const res = await fetch('http://127.0.0.1:8000/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: targetUrl })
      });

      if (!res.ok) {
        // FIX Bug #2: Server merespons tapi dengan status HTTP error
        throw new Error(`Server merespons dengan status ${res.status}`);
      }

      const data = await res.json();
      if (data.status === 'started' || data.status === 'already_running') {
        setStatus('completed');
      } else {
        // Backend merespons tapi status tidak dikenali
        throw new Error(data.message || 'Respons backend tidak dikenali.');
      }
    } catch (error) {
      // FIX Bug #2: Jangan set 'completed' saat error. Tampilkan error yang jujur.
      console.error('Failed to start stream:', error);
      setStatus('error');
    }
  }

  // Stop Analysis Handler
  const handleStopSim = async () => {
    // Tampilkan loading di header sebentar sebelum pindah ke hasil
    setStatus('processing');
    try {
      // 1. Hentikan backend terlebih dahulu
      await fetch('http://127.0.0.1:8000/api/stop', { method: 'POST' });

      // 2. Ambil data CSV DULU sebelum berpindah ke view 'analyzing_results'
      //    Ini mencegah race condition dimana dashboard sudah render tapi data masih null
      try {
        const res = await fetch('http://127.0.0.1:8000/api/logs/latest');
        const data = await res.json();
        if (data.status === 'success' && Array.isArray(data.data)) {
          setAnalyticsData(data.data);
        } else {
          // Tetap set data kosong agar dashboard tidak loading selamanya
          setAnalyticsData([]);
        }
      } catch (err) {
        console.error("Gagal mengambil data CSV:", err);
        setAnalyticsData([]);
      }

      // 3. Baru sekarang pindah ke view laporan — data sudah siap
      setStatus('analyzing_results');

    } catch (error) {
      console.error('Failed to stop stream', error);
      setStatus('idle');
    }
  }

  // Reset ke idle dari state error
  const handleReset = () => {
    setStatus('idle');
  }

  // Polling untuk live traffic status saat stream aktif
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (status === 'completed') {
      interval = setInterval(async () => {
        try {
          const res = await fetch('http://127.0.0.1:8000/api/traffic/status');
          if (res.ok) {
            const data = await res.json();
            // FIX Bug #4: Hanya update jika data valid dari backend
            if (data && data.vehicles) {
              setVehiclesCount(data.vehicles);
              setCondition(data.condition);
            }
          }
        } catch (error) {
          console.error("Error fetching traffic status:", error);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [status, setVehiclesCount, setCondition]);

  // Apakah ada data kendaraan yang nyata (bukan semua nol dari inisiasi)
  const hasLiveData = vehiclesCount.mobil > 0 || vehiclesCount.motor > 0 ||
    vehiclesCount.bis > 0 || vehiclesCount.truk > 0;

  return (
    <Card ref={containerRef} className="border-border/50 bg-card/50 backdrop-blur overflow-hidden group">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl">
          Live Monitoring
        </CardTitle>
        <div className="flex gap-2">
          {status === 'completed' && (
            <>
              <Badge variant="outline" className="animate-pulse bg-red-500/10 text-red-500 border-red-500/20">
                ● LIVE
              </Badge>
              <Button variant="destructive" size="sm" onClick={handleStopSim} className="h-6 text-xs px-2">
                Selesai & Lihat Laporan
              </Button>
            </>
          )}
          {status === 'analyzing_results' && (
            <Button variant="secondary" size="sm" onClick={() => setStatus('idle')} className="h-6 text-xs px-2">
              <XCircle className="w-3 h-3 mr-1" /> Tutup Laporan
            </Button>
          )}
          {status === 'error' && (
            <Button variant="outline" size="sm" onClick={handleReset} className="h-6 text-xs px-2 border-red-500/40 text-red-400">
              <RefreshCw className="w-3 h-3 mr-1" /> Coba Lagi
            </Button>
          )}
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={toggleFullScreen}><Maximize2 className="h-4 w-4"/></Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative aspect-video bg-black flex flex-col items-center justify-center border-y border-border">

          {/* STATE: IDLE — belum ada sumber dipilih */}
          {status === 'idle' && (
            <>
              <Play className="h-16 w-16 text-white/50 mb-4" />
              <p className="text-white/50 text-sm text-center px-4">
                {activeSource === 'CCTV'
                  ? (activeCctvUrl
                    ? `Siap memulai analisis CCTV: ${activeCctvUrl.substring(0, 50)}...`
                    : 'Masukkan URL CCTV atau pilih dari tabel untuk memulai monitoring.')
                  : (uploadedFileUrl
                    ? 'File video berhasil dipilih. Tekan "Mulai Analisis" untuk memproses.'
                    : 'Upload file video terlebih dahulu di bagian "Pilih Sumber Video".')}
              </p>
              <div className="absolute bottom-4 left-4 z-20">
                <Button
                  onClick={handleStartSim}
                  size="sm"
                  variant="secondary"
                  className="backdrop-blur bg-white/10 text-white hover:bg-white/20 disabled:opacity-40 disabled:cursor-not-allowed"
                  disabled={!isReady}
                  title={!isReady ? 'Pilih sumber video terlebih dahulu' : 'Mulai analisis YOLOv8'}
                >
                  Mulai Analisis YOLOv8
                </Button>
              </div>
            </>
          )}

          {/* STATE: PROCESSING — sedang menghubungkan ke backend */}
          {status === 'processing' && (
            <>
              <Loader2 className="h-16 w-16 text-primary animate-spin mb-4" />
              <p className="text-white">Menghubungkan ke pipeline AI...</p>
              <p className="text-white/50 text-xs mt-1">Memuat model YOLOv8 dan membuka stream video</p>
            </>
          )}

          {/* STATE: COMPLETED — stream video aktif dari backend */}
          {status === 'completed' && (
            <img
              src="http://127.0.0.1:8000/api/video_feed"
              alt="Live Video Stream"
              className="w-full h-full object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
                const parent = (e.target as HTMLImageElement).parentElement;
                if (parent && !parent.querySelector('.error-msg')) {
                  const msg = document.createElement('div');
                  msg.className = 'error-msg text-red-400 text-center p-4 text-sm';
                  msg.innerHTML = '⚠️ Stream video gagal dimuat.<br/>Pastikan backend uvicorn sudah berjalan di port 8000.';
                  parent.appendChild(msg);
                }
              }}
            />
          )}

          {/* STATE: ERROR — backend tidak bisa dihubungi */}
          {status === 'error' && (
            <div className="flex flex-col items-center gap-3 text-center px-8">
              <AlertCircle className="h-14 w-14 text-red-500 mb-2" />
              <p className="text-red-400 font-semibold text-lg">Gagal Terhubung ke Backend</p>
              <p className="text-white/50 text-sm leading-relaxed">
                Tidak dapat terhubung ke server FastAPI di <code className="text-primary">http://127.0.0.1:8000</code>.<br />
                Pastikan backend sudah dijalankan dengan perintah:<br />
                <code className="text-green-400 text-xs">uvicorn backend.main:app --reload</code>
              </p>
              <Button variant="outline" size="sm" onClick={handleReset} className="mt-2 border-red-500/30 text-red-400 hover:bg-red-500/10">
                <RefreshCw className="w-3 h-3 mr-2" /> Reset & Coba Lagi
              </Button>
            </div>
          )}

          {/* STATE: ANALYZING RESULTS — tampilkan dashboard setelah selesai */}
          {status === 'analyzing_results' && (
            <AnalyticsDashboard />
          )}
        </div>

        {/* KPI Row — FIX Bug #4: tampilkan loading dots saat belum ada data real */}
        <div className="grid grid-cols-4 divide-x divide-border border-b border-border">
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-foreground">
              {status === 'completed' && !hasLiveData
                ? <span className="text-muted-foreground text-base animate-pulse">---</span>
                : vehiclesCount.mobil}
            </div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Mobil</div>
          </div>
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-yellow-500">
              {status === 'completed' && !hasLiveData
                ? <span className="text-muted-foreground text-base animate-pulse">---</span>
                : vehiclesCount.motor}
            </div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Motor</div>
          </div>
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-green-500">
              {status === 'completed' && !hasLiveData
                ? <span className="text-muted-foreground text-base animate-pulse">---</span>
                : vehiclesCount.bis}
            </div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Bus</div>
          </div>
          <div className="p-4 text-center hover:bg-muted/50 transition-colors">
            <div className="text-2xl font-bold text-red-500">
              {status === 'completed' && !hasLiveData
                ? <span className="text-muted-foreground text-base animate-pulse">---</span>
                : vehiclesCount.truk}
            </div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">Truk</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
