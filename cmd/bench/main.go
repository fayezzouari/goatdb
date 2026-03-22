package main

import (
	"flag"
	"fmt"
	"html/template"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/index"
)

var (
	flagDim     = flag.Int("dim", 128, "vector dimension")
	flagTopK    = flag.Int("k", 10, "top-K for recall / search")
	flagQueries = flag.Int("queries", 500, "number of query vectors")
	flagOut     = flag.String("out", "bench.html", "output HTML file")
)

var datasetSizes = []int{1_000, 5_000, 10_000}

type result struct {
	index     string
	n         int
	buildMs   float64
	recallAt  float64
	qps       float64
	p50Us     float64
	p95Us     float64
	p99Us     float64
}

func randVec(dim int) core.Vector {
	emb := make([]float32, dim)
	for i := range emb {
		emb[i] = rand.Float32()*2 - 1
	}
	return core.Vector{Embeddings: emb}
}

func idFor(i int) string { return fmt.Sprintf("v%08d", i) }

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p / 100.0)
	return sorted[idx]
}

func groundTruth(vecs []core.Vector, query core.Vector, k int, metric core.DistanceMetric) map[string]struct{} {
	flat := index.NewFlatIndex(len(query.Embeddings), metric)
	for i, v := range vecs {
		flat.AddVector(idFor(i), v)
	}
	results := flat.Search(query, k)
	set := make(map[string]struct{}, len(results))
	for _, r := range results {
		set[r.Id] = struct{}{}
	}
	return set
}

type indexFactory struct {
	name    string
	build   func(dim int) core.Index
	trained bool
}

func factories(dim int) []indexFactory {
	return []indexFactory{
		{
			name:  "Flat",
			build: func(d int) core.Index { return index.NewFlatIndex(d, core.Euclidean) },
		},
		{
			name:  "LSH",
			build: func(d int) core.Index { return index.NewLSHIndex(d, 20, 8, core.Euclidean) },
		},
		{
			name: "HNSW",
			build: func(d int) core.Index {
				return index.NewHNSWIndex(d, 16, 200, 128, core.Euclidean)
			},
		},
		{
			name: "IVF",
			build: func(d int) core.Index {
				return index.NewIVFIndex(d, 16, 8, core.Euclidean)
			},
			trained: true,
		},
	}
}

func benchOne(f indexFactory, vecs []core.Vector, queries []core.Vector, k int) result {
	dim := len(vecs[0].Embeddings)
	n := len(vecs)

	// Build
	t0 := time.Now()
	idx := f.build(dim)
	if f.trained {
		if t, ok := idx.(core.Trainable); ok {
			t.Train(vecs)
		}
	}
	for i, v := range vecs {
		idx.AddVector(idFor(i), v)
	}
	buildMs := float64(time.Since(t0).Milliseconds())

	for i := 0; i < 2 && i < len(queries); i++ {
		idx.Search(queries[i], k)
	}

	latencies := make([]float64, len(queries))
	var totalRecall float64
	for qi, q := range queries {
		gt := groundTruth(vecs, q, k, core.Euclidean)

		ts := time.Now()
		res := idx.Search(q, k)
		latencies[qi] = float64(time.Since(ts).Nanoseconds()) / 1e3

		hits := 0
		for _, r := range res {
			if _, ok := gt[r.Id]; ok {
				hits++
			}
		}
		if len(gt) > 0 {
			totalRecall += float64(hits) / float64(len(gt))
		}
	}

	sort.Float64s(latencies)
	avgLatencyUs := percentile(latencies, 50)
	qps := 1e6 / avgLatencyUs

	return result{
		index:    f.name,
		n:        n,
		buildMs:  buildMs,
		recallAt: totalRecall / float64(len(queries)) * 100,
		qps:      qps,
		p50Us:    percentile(latencies, 50),
		p95Us:    percentile(latencies, 95),
		p99Us:    percentile(latencies, 99),
	}
}

func runAll(dim, k, nQueries int) []result {
	queries := make([]core.Vector, nQueries)
	for i := range queries {
		queries[i] = randVec(dim)
	}

	var results []result
	for _, n := range datasetSizes {
		fmt.Printf("dataset n=%d\n", n)
		vecs := make([]core.Vector, n)
		for i := range vecs {
			vecs[i] = randVec(dim)
		}
		for _, f := range factories(dim) {
			fmt.Printf("  %-6s ...", f.name)
			r := benchOne(f, vecs, queries, k)
			fmt.Printf(" recall=%.1f%% qps=%.0f p50=%.0fµs\n", r.recallAt, r.qps, r.p50Us)
			results = append(results, r)
		}
	}
	return results
}

const htmlTmpl = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GoatDB — ANN Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;padding:24px}
h1{font-size:1.5rem;font-weight:700;margin-bottom:4px}
p.sub{color:#94a3b8;font-size:.85rem;margin-bottom:32px}
h2{font-size:1rem;font-weight:600;color:#94a3b8;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:24px}
.card{background:#1e2130;border:1px solid #2d3148;border-radius:12px;padding:20px}
canvas{max-height:280px}
.meta{font-size:.75rem;color:#64748b;margin-top:8px}
</style>
</head>
<body>
<h1>GoatDB — ANN Benchmark Report</h1>
<p class="sub">dim={{.Dim}} &nbsp;|&nbsp; top-K={{.TopK}} &nbsp;|&nbsp; queries={{.Queries}} &nbsp;|&nbsp; {{.Date}}</p>

<div class="grid">

<div class="card">
<h2>Recall@{{.TopK}} (%)</h2>
<canvas id="recallChart"></canvas>
<p class="meta">Higher is better. Measured against exact flat-index ground truth.</p>
</div>

<div class="card">
<h2>QPS (queries / second)</h2>
<canvas id="qpsChart"></canvas>
<p class="meta">Higher is better. Derived from median per-query latency.</p>
</div>

<div class="card">
<h2>P50 Latency (µs)</h2>
<canvas id="p50Chart"></canvas>
<p class="meta">Lower is better.</p>
</div>

<div class="card">
<h2>P95 / P99 Latency (µs)</h2>
<canvas id="tailChart"></canvas>
<p class="meta">Lower is better.</p>
</div>

<div class="card">
<h2>Build Time (ms)</h2>
<canvas id="buildChart"></canvas>
<p class="meta">Lower is better. Time to insert all vectors into the index.</p>
</div>

<div class="card">
<h2>Recall vs QPS (n={{.LastN}})</h2>
<canvas id="tradeoffChart"></canvas>
<p class="meta">Upper-right is better. Each point is one index type.</p>
</div>

</div>

<script>
const COLORS = {
  Flat:  '#60a5fa',
  LSH:   '#34d399',
  HNSW:  '#f472b6',
  IVF:   '#fbbf24',
};
const SIZES  = {{.SizesJSON}};
const IDXS   = {{.IdxsJSON}};
const DATA   = {{.DataJSON}};

function col(idx){ return COLORS[idx] || '#888' }

function val(idx, n, field){
  const r = DATA.find(d => d.index===idx && d.n===n);
  return r ? r[field] : 0;
}

const barOpts = (title) => ({
  responsive:true,
  plugins:{legend:{labels:{color:'#cbd5e1'}},tooltip:{mode:'index'}},
  scales:{
    x:{ticks:{color:'#94a3b8'},grid:{color:'#2d3148'}},
    y:{ticks:{color:'#94a3b8'},grid:{color:'#2d3148'},title:{display:true,text:title,color:'#94a3b8'}}
  }
});

function makeBarChart(id, field, title){
  const labels = SIZES.map(n => n>=1000 ? n/1000+'k' : ''+n);
  const datasets = IDXS.map(idx => ({
    label: idx,
    data: SIZES.map(n => val(idx, n, field)),
    backgroundColor: col(idx)+'cc',
    borderColor: col(idx),
    borderWidth:1,
    borderRadius:4,
  }));
  new Chart(document.getElementById(id), {
    type:'bar', data:{labels, datasets}, options: barOpts(title)
  });
}

makeBarChart('recallChart', 'recallAt', 'Recall (%)');
makeBarChart('qpsChart',    'qps',      'QPS');
makeBarChart('p50Chart',    'p50Us',    'µs');
makeBarChart('buildChart',  'buildMs',  'ms');

(function(){
  const lastN = {{.LastN}};
  const labels = IDXS;
  const p95 = IDXS.map(idx => val(idx, lastN, 'p95Us'));
  const p99 = IDXS.map(idx => val(idx, lastN, 'p99Us'));
  new Chart(document.getElementById('tailChart'), {
    type:'bar',
    data:{
      labels,
      datasets:[
        {label:'P95',data:p95,backgroundColor:'#818cf8cc',borderColor:'#818cf8',borderWidth:1,borderRadius:4},
        {label:'P99',data:p99,backgroundColor:'#f43f5ecc',borderColor:'#f43f5e',borderWidth:1,borderRadius:4},
      ]
    },
    options: barOpts('µs')
  });
})();

(function(){
  const lastN = {{.LastN}};
  const datasets = IDXS.map(idx => ({
    label: idx,
    data: [{x: val(idx,lastN,'recallAt'), y: val(idx,lastN,'qps')}],
    backgroundColor: col(idx)+'cc',
    borderColor: col(idx),
    pointRadius: 10,
    pointHoverRadius: 13,
  }));
  new Chart(document.getElementById('tradeoffChart'), {
    type:'scatter',
    data:{datasets},
    options:{
      responsive:true,
      plugins:{legend:{labels:{color:'#cbd5e1'}}},
      scales:{
        x:{title:{display:true,text:'Recall (%)',color:'#94a3b8'},ticks:{color:'#94a3b8'},grid:{color:'#2d3148'}},
        y:{title:{display:true,text:'QPS',color:'#94a3b8'},ticks:{color:'#94a3b8'},grid:{color:'#2d3148'}},
      }
    }
  });
})();
</script>
</body>
</html>`

type tmplData struct {
	Dim      int
	TopK     int
	Queries  int
	Date     string
	SizesJSON template.JS
	IdxsJSON  template.JS
	DataJSON  template.JS
	LastN    int
}

func toJSON(v any) template.JS {
	switch x := v.(type) {
	case []int:
		parts := make([]string, len(x))
		for i, n := range x {
			parts[i] = fmt.Sprintf("%d", n)
		}
		return template.JS("[" + strings.Join(parts, ",") + "]")
	case []string:
		parts := make([]string, len(x))
		for i, s := range x {
			parts[i] = `"` + s + `"`
		}
		return template.JS("[" + strings.Join(parts, ",") + "]")
	case []result:
		var sb strings.Builder
		sb.WriteString("[")
		for i, r := range x {
			if i > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, `{"index":%q,"n":%d,"buildMs":%.2f,"recallAt":%.2f,"qps":%.2f,"p50Us":%.2f,"p95Us":%.2f,"p99Us":%.2f}`,
				r.index, r.n, r.buildMs, r.recallAt, r.qps, r.p50Us, r.p95Us, r.p99Us)
		}
		sb.WriteString("]")
		return template.JS(sb.String())
	}
	return template.JS("null")
}

func main() {
	flag.Parse()

	dim := *flagDim
	k := *flagTopK
	nQ := *flagQueries
	out := *flagOut

	fmt.Printf("GoatDB benchmark: dim=%d k=%d queries=%d\n\n", dim, k, nQ)

	results := runAll(dim, k, nQ)

	seen := map[string]bool{}
	var idxNames []string
	for _, r := range results {
		if !seen[r.index] {
			seen[r.index] = true
			idxNames = append(idxNames, r.index)
		}
	}
	lastN := datasetSizes[len(datasetSizes)-1]

	td := tmplData{
		Dim:       dim,
		TopK:      k,
		Queries:   nQ,
		Date:      time.Now().Format("2006-01-02 15:04 UTC"),
		SizesJSON: toJSON(datasetSizes),
		IdxsJSON:  toJSON(idxNames),
		DataJSON:  toJSON(results),
		LastN:     lastN,
	}

	f, err := os.Create(out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create %s: %v\n", out, err)
		os.Exit(1)
	}
	defer f.Close()

	t := template.Must(template.New("bench").Parse(htmlTmpl))
	if err := t.Execute(f, td); err != nil {
		fmt.Fprintf(os.Stderr, "render: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nReport written to %s\n", out)
}
