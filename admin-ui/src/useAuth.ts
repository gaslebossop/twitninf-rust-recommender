import { useCallback, useState } from 'react'

const STORAGE_KEY = 'nr_admin_key'

export function useAuth() {
  const [key, setKey] = useState<string>(() => localStorage.getItem(STORAGE_KEY) ?? '')

  const signIn = useCallback((value: string) => {
    localStorage.setItem(STORAGE_KEY, value)
    setKey(value)
  }, [])

  const signOut = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY)
    setKey('')
  }, [])

  return { key, isAuthenticated: key.length > 0, signIn, signOut }
}
