import numpy as np
import matplotlib.pyplot as plt
from contourpy.util.data import simple


def analyze(loan, apr, monthly_payment):
    r = apr / 12
    balance = loan
    total_paid = 0
    total_interest = 0
    months = 0

    balances = []
    interests = []
    totals = []

    while balance > 0:
        interest = balance * r
        principle_payment = monthly_payment - interest

        if principle_payment <= 0:
            return None
        if principle_payment > balance:
            principle_payment = balance
            payment_actual = interest + principle_payment

        else:
            payment_actual = monthly_payment

        balance -= principle_payment
        total_paid += payment_actual
        total_interest += interest
        months += 1

        balances.append(balance)
        interests.append(interest)
        totals.append(total_paid)


    return {
        "months": months,
        "total_paid": total_paid,
        "total_interest": total_interest,
        "interests": np.array(interests),
        "balance": np.array(balance),
        "total": np.array(totals)
    }



if __name__ == '__main__':
    loan = 20000
    apr = 7 / 100
    months = 84
    payment_amt = np.linspace(200, 500, 10)
    months_ar = np.linspace(0,150,151)

    fig, ax = plt.subplots(3, 1, constrained_layout=True)
    ax[0].set_title(f"Payment Options for a ${loan} Loan")
    ax[0].set_xlabel('Months')
    ax[0].set_ylabel('Total Paid [$]')

    results = {"total_paid": [],
               'total_months': []}
    for p in payment_amt:
        res = analyze(loan, apr, p)
        results['total_paid'].append(res['total_paid'])
        results['total_months'].append(res['months'])
        if res is not None:
            ax[0].plot(np.linspace(0, len(res["total"]), len(res['total'])), res['total'], label=f"${round(p, 2)}")

    # ax[0].legend(title="Payment")
    ax[0].grid(True)
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax[0].grid(which='minor', linestyle=':', linewidth='0.25', color='black', alpha=0.2)


    ax[1].plot(payment_amt, results['total_paid'])
    ax[1].set_xlabel("Payment $")
    ax[1].set_ylabel("Total Paid [$]")
    ax[1].grid(True)
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax[1].grid(which='minor', linestyle=':', linewidth='0.25', color='black', alpha=0.2)

    ax[2].plot(payment_amt, results['total_months'])
    ax[2].set_xlabel("Payment $")
    ax[2].set_ylabel("Total Months")
    ax[2].grid(True)
    ax[2].minorticks_on()
    ax[2].grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax[2].grid(which='minor', linestyle=':', linewidth='0.25', color='black', alpha=0.2)

plt.show()

