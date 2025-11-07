import React, { useState, useEffect, useRef } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  Legend,
} from 'recharts';

// Example single-file React component for previewing an evolutionary algorithm UI.
// Uses Tailwind for layout. Expects `recharts` to be available in the environment.

export default function EvolutionaryDashboard() {
  // --- Sample dataset and state ---
  const [population, setPopulation] = useState(generatePopulation(100));
  const [fitnessBins, setFitnessBins] = useState([]);
  const [lineages, setLineages] = useState(generateLineages(6));
  const [selectedIndividual, setSelectedIndividual] = useState(null);

  // Genome editor state
  const [genomeString, setGenomeString] = useState(population[0]?.genome || '');
  const [mutationRate, setMutationRate] = useState(0.05);
  const [crossoverRate, setCrossoverRate] = useState(0.5);

  useEffect(() => {
    updateFitnessBins(population);
  }, [population]);

  // --- Helpers & simulation primitives ---
  function generatePopulation(n) {
    const pop = Array.from({ length: n }).map((_, i) => {
      const genome = randomGenome(32);
      return {
        id: `ind-${i}`,
        genome,
        fitness: genomeFitness(genome),
        parent: null,
        bornAt: Date.now() - Math.floor(Math.random() * 1000000),
      };
    });
    return pop;
  }

  function generateLineages(count) {
    // create simple lineage trees with ids and children
    const roots = [];
    for (let i = 0; i < count; i++) {
      roots.push({
        id: `L-${i}`,
        name: `Lineage ${i + 1}`,
        children: buildChildren(i, 3),
      });
    }
    return roots;
  }

  function buildChildren(seed, depth) {
    if (depth === 0) return [];
    const c = [];
    const count = 1 + (seed % 3);
    for (let i = 0; i < count; i++) {
      c.push({ id: `${seed}-${depth}-${i}`, name: `N${depth}${i}`, children: buildChildren(seed + i, depth - 1) });
    }
    return c;
  }

  function randomGenome(length) {
    const chars = '01';
    return Array.from({ length }).map(() => chars[Math.random() > 0.5 ? 1 : 0]).join('');
  }

  function genomeFitness(genome) {
    // toy fitness: proportion of 1s plus small random
    const ones = genome.split('').filter((c) => c === '1').length;
    return +(ones / genome.length + Math.random() * 0.08).toFixed(3);
  }

  function updateFitnessBins(pop) {
    const bins = 10;
    const counts = Array.from({ length: bins }).map((_, i) => ({ bin: `${i}`, count: 0, mid: (i + 0.5) / bins }));
    pop.forEach((ind) => {
      const idx = Math.min(Math.floor(ind.fitness * bins), bins - 1);
      counts[idx].count += 1;
    });
    setFitnessBins(counts);
  }

  // --- UI actions ---
  function runEvolutionStep() {
    // simple selection + mutation step for demo
    const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
    const survivors = sorted.slice(0, Math.max(2, Math.floor(sorted.length * (1 - 0.3))));

    const children = [];
    while (children.length + survivors.length < population.length) {
      const a = sampleWeighted(survivors);
      const b = sampleWeighted(survivors);
      let childGenome = crossover(a.genome, b.genome, crossoverRate);
      childGenome = mutateGenome(childGenome, mutationRate);
      children.push({ id: `ind-${Math.random().toString(36).slice(2, 9)}`, genome: childGenome, fitness: genomeFitness(childGenome), parent: `${a.id},${b.id}`, bornAt: Date.now() });
    }

    const newPop = [...survivors, ...children];
    setPopulation(newPop);
    setLineages((prev) => updateLineageTree(prev, children));
  }

  function sampleWeighted(list) {
    // fitness-weighted random pick
    const total = list.reduce((s, x) => s + x.fitness, 0);
    let r = Math.random() * total;
    for (const it of list) {
      if ((r -= it.fitness) <= 0) return it;
    }
    return list[list.length - 1];
  }

  function crossover(g1, g2, rate) {
    if (Math.random() > rate) return Math.random() > 0.5 ? g1 : g2;
    const p = Math.floor(Math.random() * g1.length);
    return g1.slice(0, p) + g2.slice(p);
  }

  function mutateGenome(genome, rate) {
    return genome
      .split('')
      .map((c) => (Math.random() < rate ? (c === '1' ? '0' : '1') : c))
      .join('');
  }

  function updateLineageTree(prev, children) {
    // attach children as flat new nodes for demo
    const copy = JSON.parse(JSON.stringify(prev));
    children.forEach((child) => {
      copy[0].children.push({ id: child.id, name: child.id, children: [] });
    });
    return copy;
  }

  function editGenomeApply() {
    if (!selectedIndividual) return alert('Select an individual first');
    const newPop = population.map((p) => (p.id === selectedIndividual.id ? { ...p, genome: genomeString, fitness: genomeFitness(genomeString) } : p));
    setPopulation(newPop);
  }

  // --- Small lineage tree renderer ---
  function TreeSVG({ data, width = 400, nodeSize = { x: 140, y: 60 } }) {
    // flatten tree to nodes with positions (simple algorithm)
    const nodes = [];
    const links = [];
    let x = 0;
    function traverse(n, depth = 0) {
      const myX = x * nodeSize.x + 40;
      const myY = depth * nodeSize.y + 40;
      nodes.push({ id: n.id, name: n.name, x: myX, y: myY });
      if (n.children) {
        n.children.forEach((c) => {
          links.push({ source: n.id, target: c.id });
        });
        n.children.forEach((c) => {
          traverse(c, depth + 1);
        });
      }
      x += 1;
    }
    data.forEach((root) => traverse(root, 0));

    return (
      <svg width={width} height={Math.max(240, nodes.reduce((m, n) => Math.max(m, n.y), 0) + 80)}>
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </feMerge>
        </defs>
        {links.map((l) => {
          const s = nodes.find((n) => n.id === l.source);
          const t = nodes.find((n) => n.id === l.target);
          if (!s || !t) return null;
          return <line key={`${l.source}-${l.target}`} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="#9ca3af" strokeWidth={1.5} />;
        })}
        {nodes.map((n) => (
          <g key={n.id} transform={`translate(${n.x}, ${n.y})`}>
            <circle r={20} fill="#60a5fa" stroke="#1e3a8a" strokeWidth={2} />
            <text x={28} y={6} fontSize={12} fill="#0f172a">{n.name}</text>
          </g>
        ))}
      </svg>
    );
  }

  return (
    <div className="p-6 space-y-6 font-sans">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Evolutionary Algorithm — Visual Tools</h1>
        <div className="flex gap-2">
          <button
            onClick={runEvolutionStep}
            className="px-4 py-2 bg-gradient-to-r from-emerald-400 to-emerald-600 text-white rounded-lg shadow">
            Run Step
          </button>
        </div>
      </header>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="col-span-1 lg:col-span-2 space-y-4">
          <div className="bg-white p-4 rounded-xl shadow">
            <h2 className="font-semibold mb-2">Fitness Histogram</h2>
            <div style={{ height: 220 }}>
              <ResponsiveContainer>
                <BarChart data={fitnessBins} margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="mid" tickFormatter={(v) => v.toFixed(1)} />
                  <YAxis />
                  <Tooltip formatter={(val) => `${val} individuals`} />
                  <Bar dataKey="count" fill="#34d399" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white p-4 rounded-xl shadow">
            <h2 className="font-semibold mb-2">Population Overview</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium">Top Individuals</h3>
                <ul className="mt-2 space-y-1 max-h-36 overflow-auto">
                  {[...population].sort((a, b) => b.fitness - a.fitness).slice(0, 8).map((ind) => (
                    <li
                      key={ind.id}
                      onClick={() => {
                        setSelectedIndividual(ind);
                        setGenomeString(ind.genome);
                      }}
                      className="p-2 rounded hover:bg-gray-100 cursor-pointer flex justify-between items-center">
                      <div className="text-sm">{ind.id}</div>
                      <div className="text-xs text-gray-500">{ind.fitness.toFixed(3)}</div>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="text-sm font-medium">Stats</h3>
                <div className="mt-2 text-sm text-gray-700 space-y-1">
                  <div>Population: {population.length}</div>
                  <div>Avg Fitness: {(population.reduce((s, p) => s + p.fitness, 0) / population.length).toFixed(3)}</div>
                  <div>Selected: {selectedIndividual ? selectedIndividual.id : '—'}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-span-1 space-y-4">
          <div className="bg-white p-4 rounded-xl shadow">
            <h2 className="font-semibold mb-2">Lineage Tree</h2>
            <div className="overflow-auto">
              <TreeSVG data={lineages} width={380} />
            </div>
          </div>

          <div className="bg-white p-4 rounded-xl shadow">
            <h2 className="font-semibold mb-2">Genome Editor</h2>
            <div className="space-y-2">
              <div className="text-xs text-gray-600">Selected genome (editable)</div>
              <textarea
                className="w-full h-24 p-2 rounded border text-xs font-mono"
                value={genomeString}
                onChange={(e) => setGenomeString(e.target.value)}
              />

              <div className="flex gap-2 items-center text-sm">
                <label className="w-28">Mutation rate</label>
                <input type="range" min={0} max={0.2} step={0.005} value={mutationRate} onChange={(e) => setMutationRate(parseFloat(e.target.value))} />
                <div className="w-12 text-xs">{(mutationRate * 100).toFixed(1)}%</div>
              </div>

              <div className="flex gap-2 items-center text-sm">
                <label className="w-28">Crossover</label>
                <input type="range" min={0} max={1} step={0.05} value={crossoverRate} onChange={(e) => setCrossoverRate(parseFloat(e.target.value))} />
                <div className="w-12 text-xs">{(crossoverRate * 100).toFixed(0)}%</div>
              </div>

              <div className="flex gap-2">
                <button onClick={editGenomeApply} className="px-3 py-1 bg-blue-500 text-white rounded">Apply to selected</button>
                <button onClick={() => { setGenomeString(randomGenome(32)); }} className="px-3 py-1 bg-gray-200 rounded">Randomize</button>
                <button onClick={() => { setPopulation(generatePopulation(100)); }} className="px-3 py-1 bg-amber-400 rounded">Reset Pop</button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="text-xs text-gray-500">Tip: Click an individual to edit its genome. Use Run Step to simulate one selection + reproduction cycle.</footer>
    </div>
  );
}
