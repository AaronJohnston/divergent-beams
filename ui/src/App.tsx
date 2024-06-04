import './App.css'
import GenTree from './GenTree'

const prompt = 'What is the closest star to the Sun?'

function App() {

  return (
    <div className="App-root">
      <header className="App-header">Phuzz</header>
      <GenTree prompt={prompt}></GenTree>
    </div>
  )
}

export default App
