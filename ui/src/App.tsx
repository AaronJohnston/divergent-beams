import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <header className="App-header">Phuzz</header>
      <textarea rows={30} cols={100} className="App-prompt"></textarea>
    </>
  )
}

export default App
