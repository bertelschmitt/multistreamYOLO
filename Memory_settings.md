## How much. how little memory?

Here is a table of various memory settings used by 9 separate processes running on one 11Gbyte 1080ti.

[](/screenshots/memorytests.png)

When cramming lots of processes into one card, we must navigate the rather narrow path between not having enough memory to run one process, and not exceeding the total memory available on the GPU. The minimum memory we can get away with is around 1Gbyte per process. That comes out to 9 GBytes total, and we need to allow memory for the video driver etc.
With allow_growth set to 0 (no growth) we need to hit the memory mark rather exactly. As the table shows, anything apart from memory fraction 0.05 results in a partial or total loss.<br> However, with that number dialed-in, all 9 processes run rock-solid.
With allow_growth set to 1, we receive much more flexibility. Anything between 0.05 and 0.09 runs, because allow_growth allows every process to arrange its own memory allocation to some degree. Even if the memory fraction is set way over the limit, processes run. What we gain in convenience, we pay with stability. With allow growth set to 1, processes seem to die, sometimes after hours of running. There seems to be an unstable window between 0.10 and 0.12, and surprising stability above 0.12
Please note that the memory readings differ from run to run.  
