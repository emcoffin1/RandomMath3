import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

commute = 40    # miles
days = 5
commute = commute*days / 7  # Normalized over the week to make it easier
dol_per_gal = 5.65

# 4 Runner Info
runner_ins = 75
runner_mpg_nom = 13
runner_tank_sz = 18.5   # Gallons
runner_m_bill = 0
runner_cost = 0



# New Car Info
new_ins = 200
new_mpg_nom = 37
new_tank_sz = 12.4
new_m_bill = 425
new_cost = 25500

days = np.linspace(1,365*25,365*25)
runner_act = np.zeros(len(days))
new_act = np.zeros(len(days))

def analyze(account, mpg, tank_size, insurance, bill, cost, dol_per_gal, down=0.0, interest=0.0):

    tank = tank_size
    day_count = 0
    owed = cost - down
    owed = owed + owed*interest

    for i in range(len(account)):
        # Step up the day and reset the daily cost
        daily_cost = 0
        if i==0:
            daily_cost += down

        day_count += 1

        # Remove the gas for the day
        tank -= (commute / mpg)

        if tank <= 0:
            gallons_needed = tank_size - tank
            daily_cost += (dol_per_gal * gallons_needed)
            tank = tank_size

        # Check to see if new month (assumed 30th day)
        if day_count == 30:
            # Reset counter
            day_count = 0
            # Add insurance bill
            daily_cost += insurance
            # Add car bill if applicable
            if owed > 0:
                payment = min(bill, owed)
                owed -= payment
                daily_cost += payment

            # Add random costs
            odds = random.rand()
            daily_cost += 500 if odds > 0.95 else 0

            # Reset day count
            day_count = 0

        account[i] = account[i-1] + daily_cost

    return account









if __name__ == "__main__":
    run = []
    new = []
    for i in range(25):
        runner_act = np.zeros(len(days))
        new_act = np.zeros(len(days))
        fuel_price = np.random.normal(dol_per_gal,0.5)
        runner_mpg = np.random.normal(runner_mpg_nom, 1.0)
        new_mpg = np.random.normal(new_mpg_nom, 1.0)

        run.append(analyze(runner_act, runner_mpg, runner_tank_sz, runner_ins, runner_m_bill, runner_cost, fuel_price))

        new.append(analyze(new_act, new_mpg, new_tank_sz, new_ins, new_m_bill, new_cost, fuel_price, down=5000, interest=0.07))

    run = np.mean(run, axis=0)
    new = np.mean(new, axis=0)
    # run = analyze(runner_act, runner_mpg, runner_tank_sz, runner_ins, runner_m_bill, runner_cost)
    #
    # new = analyze(new_act, new_mpg, new_tank_sz, new_ins, new_m_bill, new_cost, down=5000, interest=0.07)

    print(f"Final Cost Difference: ${(np.abs(new[-1]-run[-1])):.2f}")

    diff = run - new
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(idx) == 0:
        print("No crossover found")
    else:
        i = idx[-1]
        x1,x2 = days[i], days[i+1]
        y1, y2 = diff[i], diff[i+1]

        x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
        print(f"Worth It After:{x_cross/365:.2f} years")

    fig, ax = plt.subplots()
    ax.plot(days/365, run, label="4Runner")
    ax.plot(days/365, new, label="New Car")
    ax.legend()
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(which="major", linestyle='-', linewidth=1, alpha=1)

    ax.set_xlabel("Years")
    ax.set_ylabel("Cost [$]")
    plt.show()



