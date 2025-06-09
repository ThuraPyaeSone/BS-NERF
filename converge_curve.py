import matplotlib.pyplot as plt

# Extracted data
batches = list(range(21, 37))  # Batches 21 to 36
losses = [
    0.4776, 0.4632, 0.4681, 0.4385, 0.4201, 0.3881, 0.3842, 0.3637,
    0.3518, 0.3899, 0.3638, 0.3485, 0.3194, 0.3148, 0.3415, 0.3197
]

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(batches, losses, marker='o', linestyle='-', color='b')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss Convergence Curve')
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig('loss_convergence_curve.png', dpi=300)

# Show plot
plt.show()
