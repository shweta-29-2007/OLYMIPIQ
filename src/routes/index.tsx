import { createFileRoute } from "@tanstack/react-router";
import hero from "@/assets/hero.jpg";

const DASHBOARD_URL = "http://localhost:8501";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "OLYMPIQ — Olympic Data Analytics & Forecasts" },
      { name: "description", content: "Explore 120 years of Olympic history: medal tables, country trajectories, breakthrough nations and medal forecasts." },
      { property: "og:title", content: "OLYMPIQ — Olympic Data Analytics" },
      { property: "og:description", content: "Medal tables, country trajectories, breakthrough watch and forecasts from 1896–2016 Olympic data." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const rings = ["border-ring-blue", "border-ring-yellow", "border-ring-black", "border-ring-green", "border-ring-red"];

const stats = [
  { n: "271K", l: "Athlete-event records" },
  { n: "120", l: "Years of Games" },
  { n: "51", l: "Olympic editions" },
  { n: "230", l: "Nations tracked" },
];

const features = [
  { k: "01", t: "Medal Tables", d: "Event-aware counting — a team gold counts once, not once per athlete.", c: "bg-ring-blue" },
  { k: "02", t: "Country Trajectories", d: "Every nation's rise and fall across editions, zero-medal years included.", c: "bg-ring-yellow" },
  { k: "03", t: "Breakthrough Watch", d: "Spot nations whose recent medal slope is climbing fastest.", c: "bg-ring-black" },
  { k: "04", t: "Sport Opportunity", d: "Find the sports where a country could win more podiums.", c: "bg-ring-green" },
  { k: "05", t: "Medal Forecast", d: "Random Forest vs. persistence baseline, backtested chronologically.", c: "bg-ring-red" },
];

function Rings({ size = "h-6 w-6" }: { size?: string }) {
  return (
    <div className="flex -space-x-1.5">
      {rings.map((r) => (
        <span key={r} className={`${size} rounded-full border-2 ${r}`} />
      ))}
    </div>
  );
}

function Index() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between px-6 py-5 md:px-12">
        <div className="flex items-center gap-3">
          <Rings size="h-4 w-4" />
          <span className="font-display text-2xl tracking-widest">OLYMPIQ</span>
        </div>
        <nav className="hidden gap-8 text-sm uppercase tracking-widest text-muted-foreground md:flex">
          <a href="#features" className="hover:text-primary">Features</a>
          <a href="#method" className="hover:text-primary">Method</a>
        </nav>
      </header>

      <section className="relative flex min-h-[92vh] items-end overflow-hidden">
        <img src={hero} alt="Sprinter in an Olympic stadium at night" width={1600} height={912} className="absolute inset-0 h-full w-full object-cover" />
        <div className="absolute inset-0 bg-hero-fade" />
        <div className="relative z-10 max-w-3xl px-6 pb-20 md:px-12">
          <p className="animate-rise mb-4 text-sm uppercase tracking-[0.3em] text-primary">Olympic Data Analytics · 1896 — 2016</p>
          <h1 className="animate-rise font-display text-7xl leading-[0.9] md:text-9xl" style={{ animationDelay: "0.1s" }}>
            Read the <span className="text-gold">podium</span>.<br />Predict the next.
          </h1>
          <p className="animate-rise mt-6 max-w-xl text-lg text-muted-foreground" style={{ animationDelay: "0.2s" }}>
            Explore 120 years of Olympic results — medal tables, rising nations and data-driven forecasts in one dashboard.
          </p>
          <div className="animate-rise mt-10 flex flex-wrap gap-4" style={{ animationDelay: "0.3s" }}>
            <a href={DASHBOARD_URL} className="bg-gold rounded-sm px-8 py-4 font-display text-xl tracking-widest text-primary-foreground transition-transform hover:-translate-y-0.5">
              Launch Dashboard →
            </a>
            <a href="#features" className="rounded-sm border border-border px-8 py-4 font-display text-xl tracking-widest hover:border-primary hover:text-primary">
              Explore features
            </a>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 border-y border-border md:grid-cols-4">
        {stats.map((s) => (
          <div key={s.l} className="border-border p-8 text-center [&:not(:last-child)]:border-r">
            <div className="font-display text-6xl text-gold">{s.n}</div>
            <div className="mt-1 text-xs uppercase tracking-widest text-muted-foreground">{s.l}</div>
          </div>
        ))}
      </section>

      <section id="features" className="px-6 py-24 md:px-12">
        <h2 className="font-display text-5xl md:text-7xl">Five lanes of insight</h2>
        <div className="mt-12 grid gap-px overflow-hidden rounded-sm border border-border bg-border md:grid-cols-5">
          {features.map((f) => (
            <article key={f.k} className="group relative bg-card p-8 transition-colors hover:bg-secondary">
              <span className={`absolute inset-x-0 top-0 h-1 ${f.c} origin-left scale-x-0 transition-transform group-hover:scale-x-100`} />
              <div className="font-display text-5xl text-muted-foreground group-hover:text-primary">{f.k}</div>
              <h3 className="mt-6 font-display text-3xl tracking-wide">{f.t}</h3>
              <p className="mt-3 text-sm text-muted-foreground">{f.d}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="method" className="grid gap-12 border-t border-border px-6 py-24 md:grid-cols-2 md:px-12">
        <div>
          <Rings size="h-10 w-10" />
          <h2 className="mt-8 font-display text-5xl md:text-6xl">Honest numbers,<br /><span className="text-gold">clear limits.</span></h2>
        </div>
        <ul className="space-y-6 text-muted-foreground">
          <li><b className="text-foreground">Fair counting.</b> Medals counted once per nation, event, edition and medal type.</li>
          <li><b className="text-foreground">History respected.</b> Historic NOC codes stay distinct — no rewriting borders.</li>
          <li><b className="text-foreground">Tested forecasts.</b> Rolling chronological validation; falls back to a baseline when it's more accurate.</li>
          <li><b className="text-foreground">Source.</b> Kaggle's "120 years of Olympic history" dataset, ending at Rio 2016.</li>
        </ul>
      </section>

      <footer className="flex flex-col items-center justify-between gap-4 border-t border-border px-6 py-8 text-sm text-muted-foreground md:flex-row md:px-12">
        <span className="font-display text-xl tracking-widest text-foreground">OLYMPIQ</span>
        <a href="https://github.com/shweta-29-2007/OLYMIPIQ" className="hover:text-primary">View on GitHub</a>
      </footer>
    </main>
  );
}
