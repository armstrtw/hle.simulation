hle <- function(pop, deaths, nr.healthy, nr.resp, adj.geo,numNeigh.geo, sumNumNeigh.geo) {
    .Call("hle",pop, deaths, nr.healthy, nr.resp, adj.geo, numNeigh.geo, sumNumNeigh.geo, PACKAGE="hle.simulation")
}
