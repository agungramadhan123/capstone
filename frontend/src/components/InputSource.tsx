import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Button } from './ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Input } from './ui/input'
import { UploadCloud, Video, Plus } from 'lucide-react'
import { useRef, useState } from 'react'

export default function InputSource() {
  const activeSource = useAppStore(state => state.activeSourceType)
  const setActiveSource = useAppStore(state => state.setActiveSourceType)
  const setSelectedCam = useAppStore(state => state.setSelectedCamera)
  const customCameras = useAppStore(state => state.customCameras)
  const addCustomCamera = useAppStore(state => state.addCustomCamera)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [newCamUrl, setNewCamUrl] = useState('')
  const [newCamLocation, setNewCamLocation] = useState('')

  const handleAddCamera = () => {
    if (newCamUrl && newCamLocation) {
      addCustomCamera({
        id: `CCT-CUS-${Date.now().toString().slice(-4)}`,
        location: newCamLocation,
        status: 'Online',
        lastActive: 'Just Added',
        streamUrl: newCamUrl
      });
      setNewCamUrl('');
      setNewCamLocation('');
    }
  }

  const handleLocalFileClick = () => {
    fileInputRef.current?.click();
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      alert(`File lokal "${file.name}" berhasil dipilih! Pada integrasi penuh, file ini akan di-upload ke backend FastAPI.`);
      // TODO: Logic for uploading to FastAPI backend via fetch
    }
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur w-full">
      <CardHeader>
        <CardTitle className="text-xl">Pilih Sumber Video</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="cctv" onValueChange={(val) => setActiveSource(val as 'CCTV' | 'Upload')}>
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="cctv">Live CCTV</TabsTrigger>
            <TabsTrigger value="upload">Upload Video</TabsTrigger>
          </TabsList>
          
          <TabsContent value="cctv">
            <div className="flex flex-col gap-4 mb-4">
              <div className="flex gap-2 items-end">
                <div className="grid gap-1.5 flex-1">
                  <label className="text-sm font-medium">Nama/Lokasi CCTV</label>
                  <Input placeholder="Contoh: Pasteur arah Dago" value={newCamLocation} onChange={e => setNewCamLocation(e.target.value)} />
                </div>
                <div className="grid gap-1.5 flex-[2]">
                  <label className="text-sm font-medium">URL RTSP / Tautan Video (HTTP/MP4)</label>
                  <Input placeholder="rtsp://... atau http://..." value={newCamUrl} onChange={e => setNewCamUrl(e.target.value)} />
                </div>
                <Button onClick={handleAddCamera} disabled={!newCamLocation || !newCamUrl}>
                  <Plus className="w-4 h-4 mr-2" /> Tambah
                </Button>
              </div>
            </div>
            <div className="rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID Kamera</TableHead>
                    <TableHead>Lokasi</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Last Active</TableHead>
                    <TableHead className="text-right">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {customCameras.map((cam) => (
                    <TableRow key={cam.id}>
                      <TableCell className="font-medium">{cam.id}</TableCell>
                      <TableCell>{cam.location}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${cam.status === 'Online' ? 'bg-green-500' : 'bg-red-500'}`}></span>
                          {cam.status}
                        </div>
                      </TableCell>
                      <TableCell>{cam.lastActive}</TableCell>
                      <TableCell className="text-right">
                        <Button 
                          size="sm" 
                          variant="secondary"
                          onClick={() => setSelectedCam(cam as any)}
                          disabled={cam.status !== 'Online'}
                        >
                          <Video className="w-4 h-4 mr-2"/> Monitor
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </TabsContent>

          <TabsContent value="upload">
            <div className="flex flex-col items-center justify-center p-12 border-2 border-dashed border-border rounded-lg bg-muted/20 hover:bg-muted/40 transition-colors cursor-pointer">
              <UploadCloud className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-1">Drag & Drop Video File</h3>
              <p className="text-sm text-muted-foreground mb-4">Support MP4, AVI, MOV (Max 2GB)</p>
              
              <div className="flex gap-4">
                <input 
                  type="file" 
                  accept="video/mp4,video/x-m4v,video/*" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                <Button variant="default" onClick={handleLocalFileClick}>
                  Select Local File
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
