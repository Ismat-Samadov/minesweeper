"use client";

import { useReducer, useEffect, useRef, useState, useCallback } from "react";
import {
  GameState,
  Difficulty,
  DIFFICULTIES,
  makeEmptyBoard,
  placeMines,
  computeAdjacent,
  revealCell,
  toggleFlag,
  chordReveal,
  checkWin,
  revealAllMines,
} from "@/lib/minesweeper";
import HUD from "./HUD";
import Board from "./Board";

// ─── Reducer ──────────────────────────────────────────────────────────────────

type Action =
  | { type: "REVEAL"; row: number; col: number }
  | { type: "FLAG"; row: number; col: number }
  | { type: "CHORD"; row: number; col: number }
  | { type: "NEW_GAME" }
  | { type: "SET_DIFFICULTY"; difficulty: Difficulty }
  | { type: "TICK" };

function makeInitialState(difficulty: Difficulty): GameState {
  const { rows, cols } = DIFFICULTIES[difficulty];
  return {
    board: makeEmptyBoard(rows, cols),
    status: "idle",
    difficulty,
    flagCount: 0,
    elapsed: 0,
    startTime: null,
    losingCell: null,
  };
}

function reducer(state: GameState, action: Action): GameState {
  switch (action.type) {
    case "NEW_GAME":
      return makeInitialState(state.difficulty);

    case "SET_DIFFICULTY":
      return makeInitialState(action.difficulty);

    case "TICK":
      if (state.status !== "playing") return state;
      return { ...state, elapsed: state.elapsed + 1 };

    case "FLAG": {
      if (state.status === "won" || state.status === "lost") return state;
      const { row, col } = action;
      const cell = state.board[row][col];
      if (cell.isRevealed) return state;
      const wasIdle = state.status === "idle";
      const newBoard = toggleFlag(state.board, row, col);
      const delta = newBoard[row][col].isFlagged ? 1 : -1;
      return {
        ...state,
        board: newBoard,
        flagCount: state.flagCount + delta,
        status: wasIdle ? "playing" : state.status,
      };
    }

    case "REVEAL": {
      if (state.status === "won" || state.status === "lost") return state;
      const { row, col } = action;
      const cell = state.board[row][col];
      if (cell.isRevealed || cell.isFlagged) return state;

      let board = state.board;
      let status = state.status;

      if (status === "idle") {
        const { mines } = DIFFICULTIES[state.difficulty];
        board = placeMines(board, mines, row, col);
        board = computeAdjacent(board);
        status = "playing";
      }

      board = revealCell(board, row, col);

      if (board[row][col].isMine) {
        return {
          ...state,
          board: revealAllMines(board),
          status: "lost",
          losingCell: [row, col],
        };
      }

      const { mines } = DIFFICULTIES[state.difficulty];
      if (checkWin(board, mines)) {
        return { ...state, board, status: "won" };
      }

      return { ...state, board, status };
    }

    case "CHORD": {
      if (state.status !== "playing") return state;
      const { row, col } = action;
      const board = chordReveal(state.board, row, col);

      for (let r = 0; r < board.length; r++) {
        for (let c = 0; c < board[r].length; c++) {
          if (board[r][c].isMine && board[r][c].isRevealed) {
            return {
              ...state,
              board: revealAllMines(board),
              status: "lost",
              losingCell: [r, c],
            };
          }
        }
      }

      const { mines } = DIFFICULTIES[state.difficulty];
      if (checkWin(board, mines)) {
        return { ...state, board, status: "won" };
      }

      return { ...state, board };
    }

    default:
      return state;
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  expert: "Expert",
};

export default function Game() {
  const [state, dispatch] = useReducer(reducer, makeInitialState("beginner"));
  const [isMouseDown, setIsMouseDown] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Timer
  useEffect(() => {
    if (state.status === "playing") {
      intervalRef.current = setInterval(() => dispatch({ type: "TICK" }), 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [state.status]);

  const handleReveal = useCallback(
    (row: number, col: number) => dispatch({ type: "REVEAL", row, col }),
    []
  );
  const handleFlag = useCallback(
    (row: number, col: number) => dispatch({ type: "FLAG", row, col }),
    []
  );
  const handleChord = useCallback(
    (row: number, col: number) => dispatch({ type: "CHORD", row, col }),
    []
  );
  const handleNewGame = useCallback(() => dispatch({ type: "NEW_GAME" }), []);
  const handleSetDifficulty = useCallback(
    (difficulty: Difficulty) =>
      dispatch({ type: "SET_DIFFICULTY", difficulty }),
    []
  );

  const { mines } = DIFFICULTIES[state.difficulty];
  const minesRemaining = mines - state.flagCount;
  const isOver = state.status === "won" || state.status === "lost";

  return (
    <div className="flex flex-col items-center gap-6 py-8 px-4">
      {/* Title */}
      <div className="text-center">
        <h1 className="text-4xl font-extrabold tracking-widest text-slate-100 drop-shadow-lg uppercase">
          💣 Minesweeper
        </h1>
        <p className="text-slate-400 text-sm mt-1 tracking-wider">
          Right-click to flag · Click number to chord
        </p>
      </div>

      {/* Difficulty buttons */}
      <div
        className="flex rounded-xl overflow-hidden border border-slate-600 shadow-lg"
        role="group"
        aria-label="Difficulty"
      >
        {(["beginner", "intermediate", "expert"] as Difficulty[]).map((d) => (
          <button
            key={d}
            onClick={() => handleSetDifficulty(d)}
            className={`px-5 py-2 text-sm font-semibold transition-colors focus:outline-none ${
              state.difficulty === d
                ? "bg-blue-600 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            {DIFFICULTY_LABELS[d]}
          </button>
        ))}
      </div>

      {/* Game panel — 3-D bezel */}
      <div
        className="rounded-xl p-3 shadow-2xl"
        style={{
          background: "#9aa5b7",
          boxShadow:
            "0 8px 32px rgba(0,0,0,0.5), inset -4px -4px 0 #5a636f, inset 4px 4px 0 #d4dce8",
          maxWidth: "100%",
        }}
      >
        <HUD
          minesRemaining={minesRemaining}
          elapsed={state.elapsed}
          status={state.status}
          isMouseDown={isMouseDown}
          onNewGame={handleNewGame}
        />

        {/* Board wrapper: inset border + horizontal scroll */}
        <div
          className="rounded overflow-auto"
          style={{
            boxShadow:
              "inset 3px 3px 0 #5a636f, inset -3px -3px 0 #d4dce8",
            padding: 3,
            maxWidth: "calc(100vw - 3rem)",
          }}
        >
          <Board
            board={state.board}
            losingCell={state.losingCell}
            gameOver={isOver}
            onReveal={handleReveal}
            onFlag={handleFlag}
            onChord={handleChord}
            onCellMouseDown={() => setIsMouseDown(true)}
            onCellMouseUp={() => setIsMouseDown(false)}
          />
        </div>
      </div>

      {/* Status banner */}
      {state.status === "won" && (
        <div className="rounded-xl bg-green-900/80 border border-green-500 text-green-300 font-bold text-lg px-8 py-3 shadow-lg text-center">
          🎉 You cleared the field!
        </div>
      )}
      {state.status === "lost" && (
        <div className="rounded-xl bg-red-900/80 border border-red-500 text-red-300 font-bold text-lg px-8 py-3 shadow-lg text-center">
          💀 Boom! Better luck next time.
        </div>
      )}

      {/* Instructions footer */}
      <p className="text-slate-500 text-xs text-center mt-2">
        {isOver
          ? "Click the face to play again."
          : state.status === "idle"
          ? "Click any cell to start."
          : ""}
      </p>
    </div>
  );
}
