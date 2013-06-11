hle <- function(pop, deaths, nr.healthy, nr.resp, adj.geo,numNeigh.geo, sumNumNeigh.geo, b0,  b1,  b2,  f3,  b4,  b5,  b6,  f7, tau.b1,  tau.b2,  tau.b5,  tau.b6) {
    .Call("hle",pop, deaths, nr.healthy, nr.resp, adj.geo, numNeigh.geo, sumNumNeigh.geo, b0,  b1,  b2,  f3,  b4,  b5,  b6,  f7, tau.b1,  tau.b2,  tau.b5,  tau.b6, PACKAGE="hle.simulation")
}
