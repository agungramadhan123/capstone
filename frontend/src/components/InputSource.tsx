import { useAppStore } from '../store'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Button } from './ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Input } from './ui/input'
import { UploadCloud, Copy, Check, Play, MonitorPlay } from 'lucide-react'
import { useRef, useState } from 'react'

export default function InputSource() {
  const activeSource = useAppStore(state => state.activeSourceType)
  const setActiveSource = useAppStore(state => state.setActiveSourceType)
  const setActiveCctvUrl = useAppStore(state => state.setActiveCctvUrl)
  const activeCctvUrl = useAppStore(state => state.activeCctvUrl)
  const setUploadedFileUrl = useAppStore(state => state.setUploadedFileUrl)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedName, setUploadedName] = useState<string | null>(null)

  const [newCamUrl, setNewCamUrl] = useState('')
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null)

  // Tabel Referensi ATCS Statis
  const ATCS_CAMERAS = [
    { location: 'SP Merdeka Aceh', url: 'https://atcs-dishub.bandung.go.id:1990/MerdekaAceh/main_stream.m3u8' },
    { location: 'Paskal dari Utara', url: 'https://atcs-dishub.bandung.go.id:1990/PaskalUt/main_stream.m3u8' },
    { location: 'Paskal dari arah Barat', url: 'https://atcs-dishub.bandung.go.id:1990/PaskalBar/main_stream.m3u8' }
  ];

  const handleTestVideo = (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    // Validasi Format URL Khusus CCTV (Strict ATCS Bandung & RTSP)
    const isRTSP = /^rtsp:\/\//i.test(newCamUrl);
    const isLocalFeed = /^http:\/\/.*:8000\/api\/video_feed/i.test(newCamUrl);
    const isAtcsBandung = /atcs-dishub\.bandung\.go\.id.*\.m3u8$/i.test(newCamUrl);
    
    if (!isRTSP && !isLocalFeed && !isAtcsBandung) {
      if (/\.m3u8$/i.test(newCamUrl)) {
        alert("Akses Ditolak (Security Whitelist)!\nSistem ini dikunci secara eksklusif hanya untuk menerima stream dari server resmi ATCS Dishub Kota Bandung.");
      } else {
        alert("Format URL CCTV tidak valid!\nHarap masukkan link CCTV (rtsp://) atau link resmi ATCS Bandung (.m3u8).\n\nUntuk file rekaman .mp4 silakan gunakan tab 'Upload Video'.");
      }
      return;
    }

    if (newCamUrl) {
      setActiveCctvUrl(newCamUrl);
      document.getElementById('monitor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // FIX Bug #3: Fungsi baru untuk langsung memilih kamera dari tabel tanpa perlu copy-paste
  const handleUseCctv = (url: string, location: string) => {
    setActiveCctvUrl(url);
    setNewCamUrl(url); // Sync ke input field agar pengguna tahu URL mana yang aktif
    document.getElementById('monitor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  const handleCopy = (url: string) => {
    navigator.clipboard.writeText(url);
    setCopiedUrl(url);
    setTimeout(() => setCopiedUrl(null), 2000);
  }

  const handleLocalFileClick = () => {
    fileInputRef.current?.click();
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setIsUploading(true);
      setUploadedName(null);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch('http://127.0.0.1:8000/api/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.status === 'success') {
          setUploadedFileUrl(data.url);
          setUploadedName(file.name);
          // FIX Bug #2 terkait: otomatis scroll ke monitor setelah upload berhasil
          setTimeout(() => {
            document.getElementById('monitor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 300);
        } else {
          alert('Gagal upload: ' + data.message);
        }
      } catch (err) {
        console.error('Upload error', err);
        alert('Gagal menghubungi server. Pastikan backend FastAPI sudah berjalan di port 8000.');
      } finally {
        setIsUploading(false);
      }
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
              <form onSubmit={handleTestVideo} className="flex gap-2 items-end">
                <div className="grid gap-1.5 flex-[1]">
                  <label className="text-sm font-medium">URL RTSP / Tautan Video (m3u8/MP4)</label>
                  <Input
                    placeholder="Paste link CCTV di sini atau pilih dari tabel di bawah..."
                    value={newCamUrl}
                    onChange={e => setNewCamUrl(e.target.value)}
                    required
                  />
                  {/* Indikator URL aktif yang sedang dipantau */}
                  {activeCctvUrl && (
                    <p className="text-xs text-green-500 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block animate-pulse" />
                      URL aktif: {activeCctvUrl.length > 60 ? activeCctvUrl.substring(0, 60) + '...' : activeCctvUrl}
                    </p>
                  )}
                </div>
                <Button type="submit">
                  <Play className="w-4 h-4 mr-2" /> Test Video
                </Button>
              </form>
            </div>

            {/* FIX Bug #3: Tabel CCTV dengan tombol "Gunakan" yang langsung memilih kamera */}
            <div className="rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[180px]">Lokasi</TableHead>
                    <TableHead>Link URL</TableHead>
                    <TableHead className="text-right w-[160px]">Aksi</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {ATCS_CAMERAS.map((cam, idx) => {
                    const isActive = activeCctvUrl === cam.url;
                    return (
                      <TableRow key={idx} className={isActive ? 'bg-primary/5 border-primary/20' : ''}>
                        <TableCell className="font-medium whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            {isActive && <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />}
                            {cam.location}
                          </div>
                        </TableCell>
                        <TableCell className="font-mono text-xs break-all text-muted-foreground">{cam.url}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex gap-1 justify-end">
                            {/* FIX Bug #3: Tombol Gunakan — langsung set URL ke state dan scroll ke monitor */}
                            <Button
                              size="sm"
                              variant={isActive ? "default" : "secondary"}
                              onClick={() => handleUseCctv(cam.url, cam.location)}
                              className="h-8"
                              title="Gunakan URL ini sebagai sumber monitoring"
                            >
                              <MonitorPlay className="w-3.5 h-3.5 mr-1" />
                              {isActive ? 'Aktif' : 'Gunakan'}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleCopy(cam.url)}
                              className="h-8 w-8 p-0"
                              title="Salin URL"
                            >
                              {copiedUrl === cam.url ? (
                                <Check className="w-3.5 h-3.5 text-green-500" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </TabsContent>

          <TabsContent value="upload">
            <div className="flex flex-col items-center justify-center p-12 border-2 border-dashed border-border rounded-lg bg-muted/20 hover:bg-muted/40 transition-colors cursor-pointer"
              onClick={!isUploading ? handleLocalFileClick : undefined}
            >
              <UploadCloud className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-1">Drag & Drop Video File</h3>
              <p className="text-sm text-muted-foreground mb-4">Support MP4, AVI, MOV (Max 2GB)</p>

              {/* FIX Bug #2 terkait: Tampilkan nama file yang berhasil diupload */}
              {uploadedName && (
                <div className="mb-4 flex items-center gap-2 text-sm text-green-500 bg-green-500/10 px-3 py-1.5 rounded-full border border-green-500/20">
                  <Check className="w-4 h-4" />
                  <span className="font-medium">{uploadedName}</span>
                  <span className="text-green-500/70">siap dianalisis</span>
                </div>
              )}

              <div className="flex gap-4" onClick={e => e.stopPropagation()}>
                <input
                  type="file"
                  accept="video/mp4,video/x-m4v,video/*"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                <Button variant="default" onClick={handleLocalFileClick} disabled={isUploading}>
                  {isUploading ? (
                    <><span className="animate-spin mr-2">⏳</span> Uploading...</>
                  ) : uploadedName ? (
                    'Ganti File Video'
                  ) : (
                    'Select Local File'
                  )}
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
