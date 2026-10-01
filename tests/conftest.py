import matplotlib

# Render plots off-screen so tests never open windows (plt.show() becomes a no-op).
matplotlib.use("Agg")
