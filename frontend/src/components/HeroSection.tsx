import { motion } from 'framer-motion'
import { ArrowRight, Video } from 'lucide-react'
import { Button } from './ui/button'

export default function HeroSection() {
  return (
    <div className="relative isolate px-6 pt-14 lg:px-8">
      <div className="mx-auto max-w-4xl py-12 sm:py-24 lg:py-32 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-400">
            Smart Traffic Bandung
          </h1>
          <p className="mt-6 text-lg leading-8 text-muted-foreground">
            Platform berbasis Artificial Intelligence untuk mendeteksi, melacak, dan menghitung kendaraan dari CCTV maupun video unggahan guna membantu analisis lalu lintas secara real-time.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Button size="lg" className="rounded-full shadow-lg shadow-primary/20">
              Mulai Monitoring <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="outline" size="lg" className="rounded-full">
              Pelajari AI <Video className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </motion.div>
      </div>

      {/* Background glow effect */}
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80" aria-hidden="true">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-primary to-blue-800 opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" style={{ clipPath: 'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)' }}></div>
      </div>
    </div>
  )
}
