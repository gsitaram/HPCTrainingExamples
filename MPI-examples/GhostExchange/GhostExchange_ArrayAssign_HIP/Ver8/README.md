# Ghost Exchange: Orig2 - CPU Implementation that decouples compute on inner cells and halo cells
  
This example shows an implementation based on Ver7 that computes the solution
first on the cells that do not need to use information from the halo cells, overlaps
this computation with the halo exchange, and finally advances the cells that use 
information from the halo. 
