height = 500;
width = 1000;
//align = "justify";
//edgeColor = "input";
csv = "energy.csv"; //csv in www directory 

// locate the charting <div>
let dom = document.getElementById('chartSankey');
// locate the <svg>
let svg = document.createElement("svg");
// hide the hide chart button
btnSankeyHide = document.getElementById("btnSankeyHide");
btnSankeyHide.style.display = "none";

// function for showing sankey
function sankeyShow(){
	options = {
		"align": document.getElementById("sankeyAlign").value,
		"edgeColor": document.getElementById("edgeColor").value
	}
	svg.remove();
	svg = plotSankey(dom, csv, options);
	btnSankeyHide.style.display = "inline-block";
};

// function for hiding sankey
function sankeyHide(){
	svg.style.display = "none";
	btnSankeyHide.style.display = "none";
}

// function for loading data
async function getDataSankey(csv) {
  const links = await d3.csv(csv, d3.autoType);
  const nodes = Array.from(new Set(links.flatMap(l => [l.source, l.target])), 
  	name => ({name, category: name.replace(/ .*/, "")}));
  return {nodes, links, units: "TWh"};
}

// function to plot the sankey
async function plotSankey(dom, csv, options){

	align = options.align;
	edgeColor = options.edgeColor;

	const sankey = ({nodes, links}) => {
	  const sankey = d3.sankey()
	      .nodeId(d => d.name)
	      .nodeAlign(d3[`sankey${align[0].toUpperCase()}${align.slice(1)}`])
	      .nodeWidth(15)
	      .nodePadding(10)
	      .extent([[1, 5], [width - 1, height - 5]]);
	  return sankey({
	    nodes: nodes.map(d => Object.assign({}, d)),
	    links: links.map(d => Object.assign({}, d))
	  });
	};

	// load data
	data = await getDataSankey(csv);

	// define color/format function
	const colorPicker = d3.scaleOrdinal(d3.schemeCategory10);
	const formatPicker = d3.format(",.0f");
	const color = d => colorPicker(d.category === undefined ? d.name : d.category);
	const format = d => data.units ? `${formatPicker(d)} ${data.units}` : formatPicker(d);

	// define plot function
	const chart = (data) => {

	  const svg = d3.create("svg")
	      .attr("viewBox", [0, 0, width, height]);

	  const {nodes, links} = sankey(data);

	  svg.append("g")
	      .attr("stroke", "#000")
	    .selectAll("rect")
	    .data(nodes)
	    .join("rect")
	      .attr("x", d => d.x0)
	      .attr("y", d => d.y0)
	      .attr("height", d => d.y1 - d.y0)
	      .attr("width", d => d.x1 - d.x0)
	      .attr("fill", color)
	    .append("title")
	      .text(d => `${d.name}\n${format(d.value)}`);

	  const link = svg.append("g")
	      .attr("fill", "none")
	      .attr("stroke-opacity", 0.5)
	    .selectAll("g")
	    .data(links)
	    .join("g")
	      .style("mix-blend-mode", "multiply");

	  if (edgeColor === "path") {
	    const gradient = link.append("linearGradient")
	        .attr("id", d => (d.uid = DOM.uid("link")).id)
	        .attr("gradientUnits", "userSpaceOnUse")
	        .attr("x1", d => d.source.x1)
	        .attr("x2", d => d.target.x0);

	    gradient.append("stop")
	        .attr("offset", "0%")
	        .attr("stop-color", d => color(d.source));

	    gradient.append("stop")
	        .attr("offset", "100%")
	        .attr("stop-color", d => color(d.target));
	  }

	  link.append("path")
	      .attr("d", d3.sankeyLinkHorizontal())
	      .attr("stroke", d => edgeColor === "none" ? "#aaa"
	          : edgeColor === "path" ? d.uid 
	          : edgeColor === "input" ? color(d.source) 
	          : color(d.target))
	      .attr("stroke-width", d => Math.max(1, d.width));

	  link.append("title")
	      .text(d => `${d.source.name} \u2192 ${d.target.name}\n${format(d.value)}`);

	  svg.append("g")
	      .attr("font-family", "sans-serif")
	      .attr("font-size", 10)
	    .selectAll("text")
	    .data(nodes)
	    .join("text")
	      .attr("x", d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
	      .attr("y", d => (d.y1 + d.y0) / 2)
	      .attr("dy", "0.35em")
	      .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
	      .text(d => d.name);

	  return svg.node();
	}

	// plot sankey
	svg = chart(data);
	dom.append(svg);
	return svg;
}