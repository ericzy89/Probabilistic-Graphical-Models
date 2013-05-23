---------------- APPROXIMATE MAP INFERENCE ----------

Max Product Linear Programming(MPLP) algorithm, a dual decomposition technique to perform approximate MAP Inference

1. Kitchen.uai and office.uai are Condition random fields. We use MPLP to perform object detection in these two images.

2. 2dri.UAI.LG and 1exm.UAI.LG are two CRF's corresponding to two proteins for which we need to solve different side-chain placement problem.
   We use MPLP to find the three dimensional configuration of rotamers given this backbone structure. 



Run the file as : python MPLP.py <names.txt> <input-CRF>

