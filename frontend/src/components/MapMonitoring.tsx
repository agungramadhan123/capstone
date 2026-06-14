import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { MapPin } from 'lucide-react'

// Fix leaflet icon issue in React
import L from 'leaflet'
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const BUAH_BATU_POS: [number, number] = [-6.9452, 107.6256]

export default function MapMonitoring() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <MapPin className="h-5 w-5 text-primary" /> Peta Lokasi CCTV Bandung
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[400px] w-full bg-muted relative z-0">
          {mounted && (
            <MapContainer 
              center={BUAH_BATU_POS} 
              zoom={14} 
              scrollWheelZoom={false}
              className="h-full w-full z-0"
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              />
              <Marker position={BUAH_BATU_POS}>
                <Popup>
                  <div className="font-sans text-sm">
                    <h4 className="font-bold mb-1">Simpang Buah Batu</h4>
                    <p className="text-muted-foreground m-0">Status: <span className="text-green-500">Online</span></p>
                    <p className="text-muted-foreground m-0">Kepadatan: Sedang</p>
                  </div>
                </Popup>
              </Marker>
            </MapContainer>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
