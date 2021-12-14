module RegularizationDP

import Optim
import Zygote

function optimize(forward_model, make_realization, regularization, x0, options)
	f = x->regularization(x) + forward_model(make_realization(x))
	opt = Optim.optimize(f, x->Zygote.gradient(f, x)[1], x0, Optim.LBFGS(), options; inplace=false)
	return make_realization(opt.minimizer), opt
end

end
