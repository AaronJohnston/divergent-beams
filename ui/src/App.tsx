import './App.css'
import TreeView from './TreeView'

const levels = [
  [
    { 'content': 'Mir', 'prob': 0.4 },
    { 'content': 'Cat', 'prob': 0.2 },
    { 'content': 'Bog', 'prob': 0.2 },
  ],
  [
    { 'content': 'ror', 'prob': 0.4, 'parent': 0 },
    { 'content': 'dog', 'prob': 0.4, 'parent': 0 },
    { 'content': 'ture', 'prob': 0.4, 'parent': 1  },
    { 'content': 'sap', 'prob': 0.4, 'parent': 1, 'status': ['eliminated'] },
    { 'content': 'lant', 'prob': 0.4, 'parent': 2  },
  ],
  [
    { 'content': 'ror', 'prob': 0.4, 'parent': 0 },
    { 'content': 'dog', 'prob': 0.4, 'parent': 1 },
    { 'content': 'ture', 'prob': 0.4, 'parent': 2  },
    { 'content': 'sap', 'prob': 0.4, 'parent': 2  },
    { 'content': 'snap', 'prob': 0.4, 'parent': 4  },
    { 'content': 'lant', 'prob': 0.4, 'parent': 4  },
  ]
]

function App() {

  return (
    <>
      <header className="App-header">Phuzz</header>
      <TreeView levels={levels}></TreeView>
    </>
  )
}

export default App
