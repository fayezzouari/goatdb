import { Background } from './components/Background.tsx'
import { Nav } from './components/Nav.tsx'
import { Hero } from './components/Hero.tsx'
import { Playground } from './components/Playground.tsx'
import { DemoVideo } from './components/DemoVideo.tsx'
import { Features } from './components/Features.tsx'
import { Benchmarks } from './components/Benchmarks.tsx'
import { QuickStart } from './components/QuickStart.tsx'
import { Footer } from './components/Footer.tsx'

export default function App() {
  return (
    <>
      <Background />
      <Nav />
      <main>
        <div className="wrap hero">
          <Hero />
          <Playground />
        </div>
        <DemoVideo />
        <Features />
        <Benchmarks />
        <QuickStart />
        <Footer />
      </main>
    </>
  )
}
