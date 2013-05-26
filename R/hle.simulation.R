run.hle <- function(pop, deaths, nr.healthy, nr.resp, adj.geo,numNeigh.geo, sumNumNeigh.geo) {
    .Call("run_hle",pop, deaths, nr.healthy, nr.resp, adj.geo, numNeigh.geo, sumNumNeigh.geo, PACKAGE="hle.simulation")
}
