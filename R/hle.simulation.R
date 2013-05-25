run.hle <- function(pop, deaths, nr_healthy, nr_resp,adj_geo,numNeigh_geo,sumNumNeigh_geo) {
    .Call("run_hle",pop, deaths, nr_healthy, nr_resp,adj_geo, numNeigh_geo,sumNumNeigh_geo,PACKAGE="hle.simulation")
}
