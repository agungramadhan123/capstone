import { create } from 'zustand'

export type TrafficStatus = 'idle' | 'processing' | 'completed' | 'analyzing_results' | 'error'
export type TrafficCondition = 'Lancar' | 'Sedang' | 'Padat' | 'Sangat Padat'

// Kamera state dihapus, disederhanakan menjadi activeCctvUrl
export interface UploadHistory {
  id: string;
  filename: string;
  uploadDate: string;
  duration: string;
  status: 'Analyzed' | 'Processing' | 'Failed';
  driveLink: string;
}

interface AppState {
  activeSourceType: 'CCTV' | 'Upload';
  setActiveSourceType: (type: 'CCTV' | 'Upload') => void;
  
  uploadedFileUrl: string | null;
  setUploadedFileUrl: (url: string | null) => void;
  
  activeCctvUrl: string | null;
  setActiveCctvUrl: (url: string | null) => void;

  trafficStatus: TrafficStatus;
  setTrafficStatus: (status: TrafficStatus) => void;

  trafficCondition: TrafficCondition;
  setTrafficCondition: (cond: TrafficCondition) => void;

  vehiclesCount: { mobil: number; motor: number; bis: number; truk: number };
  setVehiclesCount: (count: { mobil: number; motor: number; bis: number; truk: number }) => void;

  analyticsData: any[] | null;
  setAnalyticsData: (data: any[] | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeSourceType: 'CCTV',
  setActiveSourceType: (type) => set({ activeSourceType: type }),

  uploadedFileUrl: null,
  setUploadedFileUrl: (url) => set({ uploadedFileUrl: url }),

  activeCctvUrl: null,
  setActiveCctvUrl: (url) => set({ activeCctvUrl: url }),

  trafficStatus: 'idle',
  setTrafficStatus: (status) => set({ trafficStatus: status }),

  trafficCondition: 'Lancar',
  setTrafficCondition: (cond) => set({ trafficCondition: cond }),

  vehiclesCount: { mobil: 0, motor: 0, bis: 0, truk: 0 },
  setVehiclesCount: (count) => set({ vehiclesCount: count }),

  analyticsData: null,
  setAnalyticsData: (data) => set({ analyticsData: data }),
}))
