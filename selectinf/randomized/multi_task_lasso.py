{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.22454383, 0.98155081, 0.91811583],\n",
       "       [0.22454383, 0.98155081, 0.91811583]])"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "#from lasso import lasso\n",
    "import numpy as np\n",
    "K =3\n",
    "n = (3,3,3)\n",
    "p =2\n",
    "beta = np.random.random((p,K))\n",
    "\n",
    "list_X = [np.random.random((n[i],p)) for i in range(K)]\n",
    "list_Y = [np.dot(list_X[i],beta[:,i]) for i in range(K)]\n",
    "list_X_validate = [np.random.random((n[i],p)) for i in range(K)]\n",
    "list_Y_validate = [np.dot(list_X[i],beta[:,i]) for i in range(K)]\n",
    "data = {i:{'Y':list_Y[i],'X': list_X[i]} for i in range(K)}\n",
    "data_cv = {i:{'Y':list_Y[i],'X': list_X[i]} for i in range(K)}\n",
    "\n",
    "sub_gradient = [0,0,0]\n",
    "\n",
    "\n",
    "def choose_lambda(data,data_cv,beta_0,num_iter=20):\n",
    "    MSE_list = []\n",
    "    beta = beta_0\n",
    "    lambda_list = [1,2,3,4,5,6,7]\n",
    "    for parameter in lambda_list:\n",
    "        \n",
    "        for iteration in range(num_iter):\n",
    "            sum_all_tasks = np.sum(np.absolute(beta), axis=1)\n",
    "            penalty_weight = 1/np.maximum(np.sqrt(sum_all_tasks),10**-10)\n",
    "            penalty = parameter*penalty_weight\n",
    "            MSE = 0\n",
    "            \n",
    "            for task in data:\n",
    "                #solution = gaussian(d[task]['X'],d[task]['Y'],feature_weights=penalty)\n",
    "                solution = [np.random.random((1,1)),np.random.random((1,1))]\n",
    "                beta[:,task] = solution[0]\n",
    "\n",
    "        MSE_per_task = [mean_squared_error(data_cv[i]['Y'],np.dot(data_cv[i]['X'], beta[:,i])) for i in data_cv]\n",
    "        MSE_list.append(np.sum(MSE_per_task))\n",
    "\n",
    "    one_SE = np.std(MSE_list)/np.sqrt(len(MSE_list))\n",
    "    min_MSE = min(MSE_list)\n",
    "    argmin = np.argmin(abs(MSE_list-min_MSE-one_SE))\n",
    "    best_lambda = lambda_list[argmin]\n",
    "    return(best_lambda)\n",
    "\n",
    "def get_solution(data,data_cv,beta_0,num_iter=20):\n",
    "    \n",
    "    Lambda = choose_lambda(data,data_cv,beta_0,num_iter)\n",
    "    \n",
    "    for iteration in range(num_iter):\n",
    "        sum_all_tasks = np.sum(np.absolute(beta), axis=1)\n",
    "        penalty_weight = 1/np.maximum(np.sqrt(sum_all_tasks),10**-10)\n",
    "        penalty = Lambda*penalty_weight\n",
    "            \n",
    "        for task in data:\n",
    "            #solution = gaussian(d[task]['X'],d[task]['Y'],feature_weights=penalty)\n",
    "            solution = [np.random.random((1,1)),np.random.random((1,1))]\n",
    "            beta[:,task] = solution[0]\n",
    "    return(beta)\n",
    "\n",
    "get_solution(data,data_cv,beta)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.19254788324560335,\n",
       " 0.18331909467450455,\n",
       " 0.1637091748536728,\n",
       " 0.6367137360011738,\n",
       " 0.2498049546335807,\n",
       " 0.23499863213906894,\n",
       " 0.5119466392749665]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "MSE_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
