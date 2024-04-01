import matplotlib.pyplot as plt

# Create a new figure with a specific size and no margins
fig = plt.figure(figsize=(1, 1), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])

# Plot a single point with the '*' marker
ax.plot(0.5, 0.5, marker='*', markersize=50, color='orange')

# Remove axes
ax.axis('off')

# Remove all margins and padding
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

# Save the figure as a PDF
plt.savefig('star_marker.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Close the figure to free up memory
plt.close(fig)
