import { useState } from 'react';
import './App.css'
import GenTree from './GenTree'

function App() {

  const [prompt, setPrompt] = useState('What is the closest star to the Sun?')

  return (
    <div className="App-root">
      <header className="App-header">
        <h1>Tree</h1>
      </header>
      <GenTree prompt={prompt} setPrompt={setPrompt}></GenTree>
    </div>
  )
}

export default App;
