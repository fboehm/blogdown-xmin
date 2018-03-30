import numpy
from numpy.random import RandomState
from numpy.linalg import cholesky as chol
from limmbo.core.vdsimple import vd_reml
from limmbo.io.input import InputData
random = RandomState(15)
N = 100
S = 1000
P = 3
snps = (random.rand(N, S) < 0.2).astype(float)
kinship = numpy.dot(snps, snps.T) / float(10)
y  = random.randn(N, P)
pheno = numpy.dot(chol(kinship), y)
pheno_ID = [ 'PID{}'.format(x+1) for x in range(P)]
samples = [ 'SID{}'.format(x+1) for x in range(N)]
datainput = InputData()
datainput.addPhenotypes(phenotypes = pheno,
phenotype_ID = pheno_ID, pheno_samples = samples)
datainput.addRelatedness(relatedness = kinship,
relatedness_samples = samples)
Cg, Cn, ptime = vd_reml(datainput, verbose=False)
Cg
Cn
ptime
