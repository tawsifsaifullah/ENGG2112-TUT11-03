import unittest,numpy as np
from models import ExactConstantFilter,fit
from run_all import plan
class Integrity(unittest.TestCase):
 def test_tiny_variance_retained(self):
  x=np.array([[25.,1e-10,2.],[25.,2e-10,3.]])
  f=ExactConstantFilter().fit(x)
  self.assertEqual(f.keep_.tolist(),[False,True,True])
  self.assertEqual(f.transform([[45.,8e-10,4.]]).shape,(1,2))
 def test_target_train_variation_retained(self):
  f=ExactConstantFilter().fit([[25.,1.],[35.,1.]])
  self.assertEqual(f.keep_.tolist(),[True,False])
 def test_model_invariance_to_source_constant(self):
  rng=np.random.default_rng(1);x=rng.normal(size=(30,3));x[:,0]=25;y=x[:,1]*.1+3
  cfg={'seed':42,'epochs':2,'force_cpu':True,'drop_constant':True}
  model=fit('mlp_small',{'weight_decay':.001},cfg,x,y)
  shifted=x.copy();shifted[:,0]=45
  np.testing.assert_array_equal(model.predict(x),model.predict(shifted))
 def test_seed_effect(self):
  rng=np.random.default_rng(1);x=rng.normal(size=(30,3));y=x[:,1]*.1+3
  preds=[]
  for seed in [42,43]:preds.append(fit('mlp_small',{'weight_decay':.001},{'seed':seed,'epochs':2,'force_cpu':True},x,y).predict(x))
  self.assertFalse(np.array_equal(*preds))
 def test_plan(self):
  jobs=plan();self.assertEqual(len(jobs),32);self.assertEqual(len(set(jobs)),32)
if __name__=='__main__':unittest.main()
