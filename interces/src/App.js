import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

const X = 1;
const O = -1;
const EMPTY = 0;

const WIN_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

function App() {
  const [board, setBoard] = useState(Array(9).fill(EMPTY));
  const [turn, setTurn] = useState(X);
  const [winner, setWinner] = useState(null);
  const [gameMode, setGameMode] = useState('human'); // 'human', 'ml', 'hybrid'
  const [models, setModels] = useState(null);
  const [history, setHistory] = useState([]);
  const [isAiThinking, setIsAiThinking] = useState(false);

  // Load models
  useEffect(() => {
    fetch('/models.json')
      .then(res => res.json())
      .then(data => setModels(data))
      .catch(err => console.error("Failed to load models:", err));
  }, []);

  const checkWinner = (b) => {
    for (let line of WIN_LINES) {
      const [a, b1, c] = line;
      if (b[a] !== EMPTY && b[a] === b[b1] && b[a] === b[c]) {
        return b[a];
      }
    }
    if (b.every(cell => cell !== EMPTY)) return 'draw';
    return null;
  };

  const getValidMoves = (b) => {
    const moves = [];
    for (let i = 0; i < 9; i++) {
      if (b[i] === EMPTY) moves.push(i);
    }
    return moves;
  };

  // Inference using Logistic Regression
  const evaluateML = useCallback((b) => {
    if (!models) return 0;
    
    // Encode board: 18 features (c0_x, c0_o, ...)
    const features = [];
    for (let cell of b) {
      features.push(cell === X ? 1 : 0);
      features.push(cell === O ? 1 : 0);
    }

    // Scale features
    const { mean, scale } = models.scaler;
    const scaled = features.map((f, i) => (f - mean[i]) / scale[i]);

    // x_wins score
    const { coef: coefX, intercept: intX } = models.lr_xwins;
    let scoreX = intX;
    for (let i = 0; i < 18; i++) scoreX += scaled[i] * coefX[i];
    const probX = 1 / (1 + Math.exp(-scoreX));

    // is_draw score
    const { coef: coefD, intercept: intD } = models.lr_draw;
    let scoreD = intD;
    for (let i = 0; i < 18; i++) scoreD += scaled[i] * coefD[i];
    const probD = 1 / (1 + Math.exp(-scoreD));

    // Evaluation function: X wants high probX, O wants low.
    // If we are evaluating for X, return score. 
    // Heuristic: win is 1, draw is 0, loss is -1.
    // Since x_wins is for perfect play, we use it as the main signals.
    return probX - (1 - probX - probD); 
  }, [models]);

  // Minimax with ML leaf evaluation
  const minimax = useCallback((b, depth, alpha, beta, isMaximizing) => {
    const win = checkWinner(b);
    if (win === X) return 1000;
    if (win === O) return -1000;
    if (win === 'draw') return 0;
    
    if (depth === 0) {
      return evaluateML(b) * 100; // Leaf evaluation
    }

    if (isMaximizing) {
      let best = -Infinity;
      for (let move of getValidMoves(b)) {
        b[move] = X;
        let score = minimax(b, depth - 1, alpha, beta, false);
        b[move] = EMPTY;
        best = Math.max(best, score);
        alpha = Math.max(alpha, score);
        if (beta <= alpha) break;
      }
      return best;
    } else {
      let best = Infinity;
      for (let move of getValidMoves(b)) {
        b[move] = O;
        let score = minimax(b, depth - 1, alpha, beta, true);
        b[move] = EMPTY;
        best = Math.min(best, score);
        beta = Math.min(beta, score);
        if (beta <= alpha) break;
      }
      return best;
    }
  }, [evaluateML]);

  const aiMove = useCallback(() => {
    if (winner || turn === X) return; // User plays X, AI plays O
    
    setIsAiThinking(true);
    setTimeout(() => {
      const moves = getValidMoves(board);
      let bestMove = -1;

      if (gameMode === 'ml') {
        // Pure ML: chose move that minimizes evaluation (O wants to minimize)
        let bestScore = Infinity;
        for (let move of moves) {
          const nextBoard = [...board];
          nextBoard[move] = O;
          const score = evaluateML(nextBoard);
          if (score < bestScore) {
            bestScore = score;
            bestMove = move;
          }
        }
      } else if (gameMode === 'hybrid') {
        // Hybrid: Minimax depth 3 + ML eval
        let bestScore = Infinity;
        for (let move of moves) {
          const nextBoard = [...board];
          nextBoard[move] = O;
          const score = minimax(nextBoard, 3, -Infinity, Infinity, true);
          if (score < bestScore) {
            bestScore = score;
            bestMove = move;
          }
        }
      }

      if (bestMove !== -1) {
        const newBoard = [...board];
        newBoard[bestMove] = O;
        setBoard(newBoard);
        setTurn(X);
        const w = checkWinner(newBoard);
        if (w) setWinner(w);
      }
      setIsAiThinking(false);
    }, 600);
  }, [board, turn, winner, gameMode, evaluateML, minimax]);

  useEffect(() => {
    if (turn === O && (gameMode === 'ml' || gameMode === 'hybrid') && !winner) {
      aiMove();
    }
  }, [turn, gameMode, winner, aiMove]);

  const handleClick = (i) => {
    if (board[i] !== EMPTY || winner || isAiThinking) return;

    const newBoard = [...board];
    newBoard[i] = turn;
    setBoard(newBoard);
    
    const w = checkWinner(newBoard);
    if (w) {
      setWinner(w);
    } else {
      setTurn(turn === X ? O : X);
    }
  };

  const resetGame = () => {
    setBoard(Array(9).fill(EMPTY));
    setTurn(X);
    setWinner(null);
    setIsAiThinking(false);
  };

  return (
    <div className="app-container">
      <div className="background-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>

      <header className="wrapper">
        <div>
          <img src="/logo_ispm.png" alt="ISPM Logo" className="ispm-logo" />
        </div>
        <div>
          <h1>Morpion <span>IA Adaptative</span></h1>
          <div className="badge">Projet ISPM — Hackathon 2026</div>
        </div>
      </header>

      <main>
        <div className="game-wrapper">
          <div className="side-panel left">
            <h3>Mode de Jeu</h3>
            <div className="mode-selector">
              <button 
                className={gameMode === 'human' ? 'active' : ''} 
                onClick={() => { setGameMode('human'); resetGame(); }}
              >
                Humain
              </button>
              <button 
                className={gameMode === 'ml' ? 'active' : ''} 
                onClick={() => { setGameMode('ml'); resetGame(); }}
              >
                IA (ML)
              </button>
              <button 
                className={gameMode === 'hybrid' ? 'active' : ''} 
                onClick={() => { setGameMode('hybrid'); resetGame(); }}
              >
                IA (Hybride)
              </button>
            </div>

            <div className="stats-box">
              <div className="stat-item">
                <span className="label">Tour:</span>
                <span className={`value ${turn === X ? 'x-color' : 'o-color'}`}>
                  {turn === X ? 'Joueur X' : 'Joueur O'}
                </span>
              </div>
              <div className="stat-item">
                <span className="label">Status:</span>
                <span className="value">
                  {isAiThinking ? 'IA réfléchit...' : 'En attente'}
                </span>
              </div>
            </div>

            <button className="reset-btn" onClick={resetGame}>Nouvelle Partie</button>
          </div>

          <div className="board-container">
            {winner && (
              <div className="winner-overlay">
                <div className="winner-content">
                  <h2>
                    {winner === 'draw' ? 'Match Nul !' : `Victoire de ${winner === X ? 'X' : 'O'} !`}
                  </h2>
                  <button onClick={resetGame}>Rejouer</button>
                </div>
              </div>
            )}
            <div className="board">
              {board.map((cell, i) => (
                <div 
                  key={i} 
                  className={`cell ${cell === X ? 'x-cell' : cell === O ? 'o-cell' : ''} ${!winner && !isAiThinking && cell === EMPTY ? 'playable' : ''}`}
                  onClick={() => handleClick(i)}
                >
                  <span className="symbol">{cell === X ? 'X' : cell === O ? 'O' : ''}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="side-panel right">
            <h3>Analyse en Temps Réel</h3>
            {models ? (
              <div className="analysis-box">
                <p>Modèle chargé: <strong>Logistic Regression</strong></p>
                <div className="prediction-bar">
                  <div className="label-row">
                    <span>X Gagne</span>
                    <span>{(evaluateML(board) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, Math.abs(evaluateML(board) * 100)))}%` }}></div>
                  </div>
                </div>
                <div className="info-text">
                  {gameMode === 'hybrid' ? 
                    "L'IA utilise Minimax à profondeur 3 avec évaluation ML des feuilles." : 
                    "L'IA utilise directement les probabilités du modèle ML."}
                </div>
              </div>
            ) : (
              <div className="loading-models">Chargement des modèles...</div>
            )}
            
            <div className="legend">
              <h4>Légende</h4>
              <div className="legend-item"><span className="dot x-bg"></span> X : Startup EdTech</div>
              <div className="legend-item"><span className="dot o-bg"></span> O : IA Adaptative</div>
            </div>
          </div>
        </div>
      </main>

      <footer>
        <p>© 2026 Startup EdTech Madagascar — Hackathon ML Pipeline</p>
      </footer>
    </div>
  );
}

export default App;
