from habitat import logger

def log_fps(count_steps, total_time):
    logger.info(f"fps: {count_steps/total_time}")

def log_metric(stats_episodes):
    num_episodes = len(stats_episodes)
    if num_episodes > 0:
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )
        logger.info("Average metric with {} finished episodes".format(num_episodes))
        text = ""
        for k, v in aggregated_stats.items():
            text + ""
            logger.info(f"Average episode {k}: {v:.4f}")