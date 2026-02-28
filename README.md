# Minesweeper

Classic Minesweeper built with Next.js, TypeScript, and Tailwind CSS.

## Features

- Three difficulty presets: Beginner (9×9, 10 mines), Intermediate (16×16, 40 mines), Expert (16×30, 99 mines)
- Safe first click — mines are placed after the first reveal, guaranteeing the clicked cell and all its neighbors are mine-free
- Flood-fill reveal for zero-adjacent cells
- Chord reveal — left-click a numbered cell to auto-reveal its unflagged neighbors when the flag count matches
- Right-click to place / remove flags
- HUD with LED-style mine counter, timer (MM:SS), and face button
- Losing cell highlighted in red; all mines revealed on loss
- Responsive layout with horizontal scroll for the Expert grid on small screens

## Stack

- [Next.js 16](https://nextjs.org/) (App Router)
- [React 19](https://react.dev/)
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/)

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Production build |
| `npm start` | Serve production build |
| `npm run lint` | Run ESLint |

## Project Structure

```
app/
  layout.tsx        # Root layout, font, background
  page.tsx          # Renders <Game />
  globals.css       # 3-D cell styles, LED display
  icon.svg          # Favicon
lib/
  minesweeper.ts    # All pure game logic and types
components/
  Game.tsx          # useReducer state, timer interval, action handlers
  HUD.tsx           # Mine counter, face button, timer
  Board.tsx         # Grid layout with overflow scroll
  Cell.tsx          # Cell rendering and click handlers
```

## Controls

| Input | Action |
|---|---|
| Left click (unrevealed) | Reveal cell |
| Right click (unrevealed) | Toggle flag |
| Left click (revealed number) | Chord reveal |
