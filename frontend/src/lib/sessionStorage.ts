import type { StateStorage } from "zustand/middleware";

const fallbackStorage = new Map<string, string>();

function browserSessionStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export const safeSessionStorage: StateStorage = {
  getItem(name) {
    return browserSessionStorage()?.getItem(name) ?? fallbackStorage.get(name) ?? null;
  },
  setItem(name, value) {
    const storage = browserSessionStorage();
    if (storage) storage.setItem(name, value);
    else fallbackStorage.set(name, value);
  },
  removeItem(name) {
    const storage = browserSessionStorage();
    if (storage) storage.removeItem(name);
    fallbackStorage.delete(name);
  },
};

export function clearSessionValue(name: string): void {
  void safeSessionStorage.removeItem(name);
}
