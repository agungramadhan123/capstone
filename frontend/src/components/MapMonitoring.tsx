import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { MapPin, Wifi, WifiOff } from 'lucide-react'
import L from 'leaflet'
import { useAppStore } from '../store'

// Fix leaflet icon issue in React
delete (L.Icon.Default.prototype as any)._getIconUrl;

// Icon default (abu-abu) untuk kamera tidak aktif
const defaultIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-grey.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// Icon biru untuk kamera yang sedang aktif dimonitor
const activeIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [30, 46],
  iconAnchor: [15, 46],
  popupAnchor: [1, -38],
  shadowSize: [41, 41],
});

// Icon merah untuk kamera yang offline
const offlineIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// Icon hijau untuk lokasi GPS pengguna
const userLocationIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// FIX Bug #6: Data kamera CCTV yang lengkap dengan koordinat GPS nyata di Bandung
interface CctvCamera {
  id: string;
  location: string;
  url: string;
  position: [number, number];
  isOnline: boolean;
  district: string;
}

const CCTV_CAMERAS: CctvCamera[] = [
  {
    id: 'buah-batu',
    location: 'Simpang Buah Batu',
    url: '',
    position: [-6.9452, 107.63352],
    isOnline: true,
    district: 'Kec. Buah Batu',
  },
  {
    id: 'merdeka-aceh',
    location: 'SP Merdeka Aceh',
    url: 'https://atcs-dishub.bandung.go.id:1990/MerdekaAceh/main_stream.m3u8',
    position: [-6.9022, 107.6152],
    isOnline: true,
    district: 'Kec. Coblong',
  },
  {
    id: 'paskal-utara',
    location: 'Paskal dari Utara',
    url: 'https://atcs-dishub.bandung.go.id:1990/PaskalUt/main_stream.m3u8',
    position: [-6.8909, 107.5988],
    isOnline: true,
    district: 'Kec. Andir',
  },
  {
    id: 'paskal-barat',
    location: 'Paskal dari arah Barat',
    url: 'https://atcs-dishub.bandung.go.id:1990/PaskalBar/main_stream.m3u8',
    position: [-6.8915, 107.5979],
    isOnline: false,
    district: 'Kec. Andir',
  },
];

// Pusat peta: titik tengah kota Bandung
const MAP_CENTER: [number, number] = [-6.9175, 107.6098];

export default function MapMonitoring() {
  const [mounted, setMounted] = useState(false)
  const [userLocation, setUserLocation] = useState<[number, number] | null>(null)

  // FIX Bug #6: Ambil data kondisi & kamera aktif dari Zustand state
  const trafficCondition = useAppStore(state => state.trafficCondition)
  const trafficStatus = useAppStore(state => state.trafficStatus)
  const activeCctvUrl = useAppStore(state => state.activeCctvUrl)
  const vehiclesCount = useAppStore(state => state.vehiclesCount)

  useEffect(() => {
    setMounted(true)
    // Meminta izin dan mendeteksi lokasi GPS perangkat
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition((position) => {
        setUserLocation([position.coords.latitude, position.coords.longitude])
      }, (error) => {
        console.error("Gagal mendapatkan lokasi GPS:", error);
      });
    }
  }, [])

  // Tentukan warna badge kondisi
  const conditionColor = {
    'Lancar': 'text-green-500',
    'Sedang': 'text-yellow-500',
    'Padat': 'text-orange-500',
    'Sangat Padat': 'text-red-500',
  }[trafficCondition] ?? 'text-green-500';

  const isMonitoringActive = trafficStatus === 'completed';

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur overflow-hidden">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-xl">
            <MapPin className="h-5 w-5 text-primary" /> Peta Lokasi CCTV Bandung
          </CardTitle>
          {/* FIX Bug #6: Legenda dinamis berdasarkan kondisi real dari Zustand */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> Lokasi Anda
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> Aktif dipantau
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-gray-400 inline-block" /> Online
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> Offline
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[420px] w-full bg-muted relative z-0">
          {mounted && (
            <MapContainer
              center={MAP_CENTER}
              zoom={13}
              scrollWheelZoom={false}
              className="h-full w-full z-0"
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
              />

              {/* FIX Bug #6: Render semua kamera CCTV dengan marker dinamis */}
              {CCTV_CAMERAS.map((cam) => {
                const isActive = isMonitoringActive && activeCctvUrl === cam.url;
                const icon = isActive ? activeIcon : (!cam.isOnline ? offlineIcon : defaultIcon);
                const totalVehicles = isActive
                  ? vehiclesCount.mobil + vehiclesCount.motor + vehiclesCount.bis + vehiclesCount.truk
                  : null;

                return (
                  <Marker key={cam.id} position={cam.position} icon={icon}>
                    <Popup>
                      <div className="font-sans text-sm min-w-[180px]">
                        <h4 className="font-bold mb-2 text-gray-800">{cam.location}</h4>
                        <div className="space-y-1">
                          <p className="text-gray-500 text-xs">{cam.district}</p>
                          {/* FIX Bug #6: Status online/offline dari data kamera */}
                          <p className="m-0 flex items-center gap-1">
                            {cam.isOnline
                              ? <><span className="text-green-600 font-medium">● Online</span></>
                              : <><span className="text-red-500 font-medium">● Offline</span></>}
                          </p>
                          {/* FIX Bug #6: Kondisi lalu lintas dari Zustand (real saat kamera aktif) */}
                          <p className="m-0 text-gray-600">
                            Kepadatan:{' '}
                            <span className={`font-semibold ${isActive ? conditionColor : 'text-gray-400'}`}>
                              {isActive ? trafficCondition : 'Tidak dipantau'}
                            </span>
                          </p>
                          {/* Tampilkan hitungan kendaraan hanya untuk kamera yang aktif dipantau */}
                          {isActive && totalVehicles !== null && totalVehicles > 0 && (
                            <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-600 space-y-0.5">
                              <p className="font-semibold text-gray-700 mb-1">Hitungan Kendaraan:</p>
                              <p>🚗 Mobil: <span className="font-medium">{vehiclesCount.mobil}</span></p>
                              <p>🏍️ Motor: <span className="font-medium">{vehiclesCount.motor}</span></p>
                              <p>🚌 Bus: <span className="font-medium">{vehiclesCount.bis}</span></p>
                              <p>🚛 Truk: <span className="font-medium">{vehiclesCount.truk}</span></p>
                            </div>
                          )}
                          {isActive && (
                            <p className="mt-1 text-xs text-blue-600 font-medium animate-pulse">
                              ● Sedang dipantau
                            </p>
                          )}
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                );
              })}

              {/* Render Marker Lokasi Pengguna Jika GPS Ditemukan */}
              {userLocation && (
                <Marker position={userLocation} icon={userLocationIcon}>
                  <Popup>
                    <div className="font-sans text-sm min-w-[120px]">
                      <h4 className="font-bold mb-1 text-green-700">📍 Lokasi Anda</h4>
                      <p className="text-gray-500 text-xs m-0">Akurasi disinkronkan dari GPS perangkat Anda.</p>
                    </div>
                  </Popup>
                </Marker>
              )}
            </MapContainer>
          )}
        </div>

        {/* FIX Bug #6: Panel ringkasan di bawah peta — dinamis dari Zustand */}
        <div className="p-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground flex-wrap gap-2">
          <span className="flex items-center gap-1.5">
            <MapPin className="w-3 h-3 text-primary" />
            {CCTV_CAMERAS.filter(c => c.isOnline).length} kamera online dari {CCTV_CAMERAS.length} total
          </span>
          {isMonitoringActive ? (
            <span className="flex items-center gap-1.5">
              <Wifi className="w-3 h-3 text-green-500" />
              Monitoring aktif — Kepadatan: <span className={`font-semibold ${conditionColor}`}>{trafficCondition}</span>
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <WifiOff className="w-3 h-3 text-muted-foreground" />
              Tidak ada monitoring aktif saat ini
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
