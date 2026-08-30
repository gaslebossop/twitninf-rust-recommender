import { useState } from 'react'
import { useAuth } from './useAuth'
import { SignIn } from './SignIn'
import { Dashboard } from './pages/Dashboard'
import { Weights } from './pages/Weights'
import { Model } from './pages/Model'
import { Moderation } from './pages/Moderation'
import { Logs } from './pages/Logs'

const SECTIONS = [
  { id: 'dashboard', label: 'Instruments' },
  { id: 'weights', label: 'Poids' },
  { id: 'model', label: 'Modèle' },
  { id: 'moderation', label: 'Modération' },
  { id: 'logs', label: 'Journal' },
] as const

type SectionId = (typeof SECTIONS)[number]['id']

export default function App() {
  const { key, isAuthenticated, signIn, signOut } = useAuth()
  const [section, setSection] = useState<SectionId>('dashboard')

  if (!isAuthenticated) {
    return <SignIn onSignIn={signIn} />
  }

  return (
    <div className="shell">
      <nav className="rail">
        <div className="rail-mark">NR</div>
        <div className="rail-nav">
          {SECTIONS.map((s) => (
            <button key={s.id} className={section === s.id ? 'rail-item active' : 'rail-item'} onClick={() => setSection(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
        <button className="rail-signout" onClick={signOut}>
          Se déconnecter
        </button>
      </nav>
      <main className="content">
        {section === 'dashboard' && <Dashboard apiKey={key} />}
        {section === 'weights' && <Weights apiKey={key} />}
        {section === 'model' && <Model apiKey={key} />}
        {section === 'moderation' && <Moderation apiKey={key} />}
        {section === 'logs' && <Logs apiKey={key} />}
      </main>
    </div>
  )
}
