from matplotlib import pyplot as plt
import numpy as np

train_files = ["./src/rl_td3/rewards_train.txt","./src/rl_a2c/rewards_train.txt"]
test_files = ["./src/rl_td3/rewards_test.txt","./src/rl_a2c/rewards_test.txt"]

def read_files(train_files,countIdx):
    alg_rewards = []
    alg_finish = []
    for file in train_files:
        count = 0
        total_rewards = []
        total_finish = []
        with open(file,"r") as f:
            rewards = []
            finish = []
            cc = 0
            for line in f:
                cc += 1
                if count >= countIdx:
                    total_rewards.append(np.mean(rewards))
                    total_finish.append(np.mean(finish)*100)
                    rewards = []
                    finish = []
                    count = 0
                split = line.split(" ")
                if split[0]=="start\n":
                    continue
                step = []
                for val in split:
                    if val=="\n":
                        continue
                    try:
                        step.append(eval(val))
                    except SyntaxError:
                        print("end of file")
                rewards.append(np.sum(step))
                if step[-1]==100:
                    finish.append(1)
                else:
                    finish.append(0)
                count += 1
        alg_rewards.append(total_rewards)
        alg_finish.append(total_finish)

    return alg_rewards,alg_finish

alg_rewards_train,alg_finish_train = read_files(train_files,100)
alg_rewards_test,alg_finish_test = read_files(test_files,10)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
fsize = 16
ax1.set_title("average reward over time",fontsize=fsize)
ax2.set_title("average runs finished over time",fontsize=fsize)

ax1.set_xlabel("learning step",fontsize=fsize)
ax1.set_ylabel("reward",fontsize=fsize)

ax2.set_xlabel("learning step",fontsize=fsize)
ax2.set_ylabel("percentage finished",fontsize=fsize)

label_list = ["td3","a2c"]
# for idx in range(len(alg_finish_train)):
#     ax1.plot(alg_rewards_train[idx],label=label_list[idx])
#     ax2.plot(alg_finish_train[idx],label=label_list[idx])
for idx in range(len(alg_finish_test)):
    ax1.plot(alg_rewards_test[idx],label=label_list[idx])
    ax2.plot(alg_finish_test[idx],label=label_list[idx])

ax1.legend(fontsize=fsize)
ax2.legend(fontsize=fsize)

ax1.tick_params(axis="both",labelsize=fsize)

plt.show()