import { create } from 'zustand'

export type TrafficStatus = 'idle' | 'processing' | 'completed'
export type TrafficCondition = 'Lancar' | 'Sedang' | 'Padat' | 'Sangat Padat'

export interface Camera {
  id: string;
  location: string;
  status: 'Online' | 'Offline';
  lastActive: string;
  streamUrl?: string;
}

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
  
  selectedCamera: Camera | null;
  setSelectedCamera: (cam: Camera | null) => void;

  customCameras: Camera[];
  addCustomCamera: (cam: Camera) => void;

  trafficStatus: TrafficStatus;
  setTrafficStatus: (status: TrafficStatus) => void;

  trafficCondition: TrafficCondition;
  setTrafficCondition: (cond: TrafficCondition) => void;

  vehiclesCount: { mobil: number; motor: number; bis: number; truk: number };
  setVehiclesCount: (count: { mobil: number; motor: number; bis: number; truk: number }) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeSourceType: 'CCTV',
  setActiveSourceType: (type) => set({ activeSourceType: type }),

  selectedCamera: null,
  setSelectedCamera: (cam) => set({ selectedCamera: cam }),

  customCameras: [
    { id: 'CCT-001', location: 'Simpang Buah Batu', status: 'Online', lastActive: 'Now', streamUrl: 'https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4' },
    { id: 'CCT-002', location: 'Buah Batu Utara', status: 'Online', lastActive: 'Now' },
    { id: 'CCT-003', location: 'Buah Batu Selatan', status: 'Offline', lastActive: '2h ago' },
  ],
  addCustomCamera: (cam) => set((state) => ({ customCameras: [...state.customCameras, cam] })),

  trafficStatus: 'idle',
  setTrafficStatus: (status) => set({ trafficStatus: status }),

  trafficCondition: 'Lancar',
  setTrafficCondition: (cond) => set({ trafficCondition: cond }),

  vehiclesCount: { mobil: 0, motor: 0, bis: 0, truk: 0 },
  setVehiclesCount: (count) => set({ vehiclesCount: count }),
}))
