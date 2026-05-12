"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time, argparse, json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import multiprocessing as mp
from functools import partial

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.utils.eff_criterion import nse
import logging, coloredlogs

def get_available_countries():
    """
    Get a list of available countries by looking at directories
    
    Returns:
        list: List of country names (directories)
    """
    data_base_dir = os.path.join(dirname, 'data')
    
    if not os.path.exists(data_base_dir):
        logger.warning(f"Data directory not found at {data_base_dir}")
        return []
    
    # Get all directories in the data folder
    return [d for d in os.listdir(data_base_dir) 
            if os.path.isdir(os.path.join(data_base_dir, d)) and d != "world"]

def sanitize_country_name(country):
    """
    Sanitize country name for use in file paths and filenames
    
    Args:
        country (str): Country name
    
    Returns:
        str: Sanitized country name
    """
    # Replace any unsafe characters
    safe_name = country.replace(' ', '_').replace('/', '_').replace('\\', '_')
    return safe_name


def process_locations_parallel(filenames, args, num_processes=None):
    """
    Process multiple locations in parallel
    
    Args:
        filenames (list): List of CSV filenames to process
        args: Command line arguments with country and year information
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    logger.info(f"Starting parallel processing with {num_processes} processes for country: {args.country}")
    
    # Create a partial function with fixed args
    process_func = partial(process_single_location, args=args)
    
    # Create a process pool and map the work
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, filenames)
    
    # Count successful processes
    successful = sum(1 for r in results if r)
    logger.info(f"Processed {successful} out of {len(filenames)} locations successfully for {args.country}")


def get_data_filenames(country=None):
    """Find all CSV files in the data directory for the specified country"""
    if country is None:
        raise ValueError("Country must be specified")
    
    data_base_dir = os.path.join(dirname, 'data')
    country_dir = os.path.join(data_base_dir, country)
    era5_dir = os.path.join(country_dir, 'era5')
    
    # Check if the directory exists
    if not os.path.exists(era5_dir):
        raise FileNotFoundError(f"Era5 directory not found: {era5_dir}")
    
    # Look for CSV files in the data directory
    csv_files = [f for f in os.listdir(era5_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {era5_dir}")
    
    return csv_files, country_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class and consolidate by country")
    parser.add_argument("--start_year", required=False, help="Specify the start year (e.g., 2019)")
    parser.add_argument("--end_year", required=False, help="Specify the end year (e.g., 2020)")
    parser.add_argument("--country", required=True, help="Specify the country name (used as data/country)")
    parser.add_argument("--list_countries", action="store_true", help="List all available countries")
    return parser.parse_args()

def list_available_countries():
    """List all available countries by checking existing directories"""
    countries = get_available_countries()
    
    if countries:
        print("\nAvailable countries:")
        for i, country in enumerate(sorted(countries), 1):
            print(f"  {i}. {country}")
        print(f"\nTotal: {len(countries)} countries\n")
    else:
        print("No country directories found in data folder")

def extract_location_details(filename):
    # Strip the file extension if present
    if filename.endswith('.csv'):
        location = filename[:-4]  # Remove .csv extension
    else:
        location = filename

    # Split the string based on underscore
    try:
        lat, long, alt = location.split('_')
        coords = [float(lat), float(long)]
        alt = float(alt)
    except ValueError:
        raise ValueError("Filename should be in the format lat_long_alt.csv (e.g., 34.216_77.606_4009.csv)")

    return coords, alt, location

def process_single_location(filename, args):
    """Process a single location's data"""
    try:
        logger.info(f"\nProcessing {filename}")
        coords, alt, location = extract_location_details(filename)

        # Build data path from country
        country_dir = os.path.join(dirname, 'data', args.country)
        
        SITE, FOLDER = config(location, start_year=args.start_year, end_year=args.end_year, 
                            alt=alt, coords=coords, datadir=country_dir)
        
        # Ensure the processed directory exists
        os.makedirs(FOLDER['output'], exist_ok=True)
        
        icestupa = Icestupa(SITE, FOLDER)
        icestupa.sim_air(test=False)
        
        logger.info(f"Completed processing {filename}")
        return True
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return False

def process_single_result(filename, country):
    """
    Process a single location's results with country information.
    
    Args:
        filename (str): CSV filename to process
        country (str): Country name
    
    Returns:
        dict: Location results with coordinates, country and metrics
    """
    try:
        # Extract location details from filename
        if filename.endswith('.csv'):
            location = filename[:-4]
        else:
            location = filename
            
        lat, lon, alt = location.split('_')
        lat = float(lat)
        lon = float(lon)
        alt = float(alt)
        
        # Get just the country name without path
        country_name = os.path.basename(country)
        
        # Build data path from country
        data_dir = os.path.join(dirname, 'data', country_name)
        
        # Read the individual results.json for this location
        location_dir = os.path.join(data_dir, location, "processed")
        results_file = os.path.join(location_dir, "results.json")
        
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return None
            
        with open(results_file, "r") as f:
            location_results = json.load(f)
        
        # Create consolidated entry
        return {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "country": country_name,
            "iceV_max": location_results["iceV_max"],
            "survival_days": location_results["survival_days"]
        }
        
    except (ValueError, FileNotFoundError, KeyError) as e:
        logger.error(f"Error processing results for {filename} in {country}: {str(e)}")
        return None

def consolidate_results(filenames, output_dir):
    """
    Consolidate results from multiple locations into a single JSON file.
    
    Args:
        filenames (list): List of CSV filenames processed
        output_dir (str): Directory to save the consolidated results
    """
    consolidated_results = []
    
    for filename in filenames:
        # Extract location details from filename
        if filename.endswith('.csv'):
            location = filename[:-4]
        else:
            location = filename
            
        try:
            lat, lon, alt = location.split('_')
            lat = float(lat)
            lon = float(lon)
            alt = float(alt)
            
            # Read the individual results.json for this location
            location_dir = os.path.join(output_dir, location, "processed")
            with open(os.path.join(location_dir, "results.json"), "r") as f:
                location_results = json.load(f)
            
            # Create consolidated entry
            result_entry = {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "iceV_max": location_results["iceV_max"],
                "survival_days": location_results["survival_days"]
            }
            
            consolidated_results.append(result_entry)
            
        except (ValueError, FileNotFoundError, KeyError) as e:
            logger.error(f"Error processing results for {filename}: {str(e)}")
            continue
    
    # Save consolidated results
    with open(os.path.join(output_dir, "consolidated_results.json"), "w") as f:
        json.dump(consolidated_results, f, indent=4, sort_keys=True)
    
    logger.info(f"Consolidated results saved to {output_dir}/consolidated_results.json")
    
    return consolidated_results

def consolidate_results_parallel(filenames, country, num_processes=None):
    """
    Consolidate results from multiple locations in parallel and save by country.
    
    Args:
        filenames (list): List of CSV filenames processed
        country (str): Country name
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
    
    Returns:
        dict: Dictionary with country as key and list of location results as value
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    logger.info(f"Starting results consolidation for country: {country}...")
    
    # Create a partial function with fixed country
    process_func = partial(process_single_result, country=country)
    
    # Create a process pool and map the work
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, filenames)
    
    # Filter out None results (from errors)
    valid_results = [r for r in results if r is not None]
    
    # Create world directory if it doesn't exist
    data_base_dir = os.path.join(dirname, 'data')
    world_dir = os.path.join(data_base_dir, "world")
    os.makedirs(world_dir, exist_ok=True)
    
    # Calculate country-level statistics
    if valid_results:
        summary_stats = {
            "total_locations": len(valid_results),
            "average_survival_days": sum(r['survival_days'] for r in valid_results) / len(valid_results),
            "max_iceV": max(r['iceV_max'] for r in valid_results),
            "min_iceV": min(r['iceV_max'] for r in valid_results),
            "average_iceV": sum(r['iceV_max'] for r in valid_results) / len(valid_results),
            "average_altitude": sum(r['altitude'] for r in valid_results) / len(valid_results)
        }
        
        country_name = os.path.basename(country)
        # Prepare country output data
        output_data = {
            "country": country_name,
            "summary_statistics": summary_stats,
            "location_results": valid_results
        }
        
        # Create safe filename
        safe_filename = country_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Save to country JSON file in world_dir
        country_file = os.path.join(world_dir, f"{safe_filename}.json")
        with open(country_file, "w") as f:
            json.dump(output_data, f, indent=4, sort_keys=True)
        
        logger.info(f"Saved {country} results to {country_file} with {len(valid_results)} locations")
        
        # Also save consolidated results in the country's directory
        country_dir = os.path.join(data_base_dir, country)
        consolidated_file = os.path.join(country_dir, "consolidated_results.json")
        with open(consolidated_file, "w") as f:
            json.dump(valid_results, f, indent=4, sort_keys=True)
        
        logger.info(f"Processed {len(valid_results)} locations for {country}")
        
        # Return a simple dictionary with just this country
        return {country_name: valid_results}
    else:
        logger.warning(f"No valid results found for {country}")
        return {}

def update_world_summary():
    """
    Update the summary file in the world directory by scanning all country JSON files
    
    Returns:
        dict: Dictionary of all countries and their summary information
    """
    # Define world directory path
    world_dir = os.path.join(dirname, 'data', 'world')
    
    # Create output directory if it doesn't exist
    os.makedirs(world_dir, exist_ok=True)
    
    # Get all JSON files in world_dir (except summary.json)
    json_files = [f for f in os.listdir(world_dir) if f.endswith('.json') and f != "summary.json"]
    
    # Dictionary to store countries and their data
    countries = {}
    total_locations = 0
    
    # Process each country file
    for json_file in json_files:
        country_path = os.path.join(world_dir, json_file)
        try:
            with open(country_path, 'r') as f:
                country_data = json.load(f)
                
            country_name = country_data.get('country', os.path.splitext(json_file)[0])
            location_count = len(country_data.get('location_results', []))
            total_locations += location_count
            
            countries[country_name] = {
                'file': json_file,
                'location_count': location_count,
                'summary_stats': country_data.get('summary_statistics', {})
            }
        except Exception as e:
            logger.error(f"Error reading country file {json_file}: {str(e)}")
    
    # Create overall summary
    overall_summary = {
        "total_countries": len(countries),
        "countries": list(countries.keys()),
        "total_locations": total_locations,
        "locations_by_country": {country: data['location_count'] for country, data in countries.items()},
        "country_stats": {country: data['summary_stats'] for country, data in countries.items()}
    }
    
    # Save overall summary
    summary_file = os.path.join(world_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=4, sort_keys=True)
    
    logger.info(f"Updated world summary with {len(countries)} countries and {total_locations} total locations")
    
    return countries

def create_country_summary_figures():
    """
    Create summary figures comparing countries
    """
    # Define world directory path
    world_dir = os.path.join(dirname, 'data', 'world')
    
    # Get all JSON files in the world directory
    json_files = [f for f in os.listdir(world_dir) if f.endswith('.json') and f != "summary.json"]
    
    countries = []
    avg_survival = []
    max_icev = []
    avg_altitude = []
    location_count = []
    
    # Extract data from each country file
    for json_file in json_files:
        with open(os.path.join(world_dir, json_file), 'r') as f:
            data = json.load(f)
            
        if 'country' in data and data['country'] != "Unknown":
            countries.append(data['country'])
            avg_survival.append(data['summary_statistics']['average_survival_days'])
            max_icev.append(data['summary_statistics']['max_iceV'])
            avg_altitude.append(data['summary_statistics']['average_altitude'])
            location_count.append(data['summary_statistics']['total_locations'])
    
    # If we have at least 2 countries, create comparative charts
    if len(countries) >= 2:
        # Sort countries by max_icev for better visualization
        sorted_indices = np.argsort(max_icev)[::-1]  # Sort in descending order
        countries = [countries[i] for i in sorted_indices]
        avg_survival = [avg_survival[i] for i in sorted_indices]
        max_icev = [max_icev[i] for i in sorted_indices]
        avg_altitude = [avg_altitude[i] for i in sorted_indices]
        location_count = [location_count[i] for i in sorted_indices]
        
        # Create comparative bar charts
        plt.figure(figsize=(12, 8))
        
        # Max iceV by country
        plt.subplot(2, 2, 1)
        plt.bar(countries, max_icev, color='skyblue')
        plt.title('Maximum Ice Volume by Country')
        plt.ylabel('Ice Volume (m³)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Average survival days by country
        plt.subplot(2, 2, 2)
        plt.bar(countries, avg_survival, color='lightgreen')
        plt.title('Average Survival Days by Country')
        plt.ylabel('Days')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Average altitude by country
        plt.subplot(2, 2, 3)
        plt.bar(countries, avg_altitude, color='salmon')
        plt.title('Average Altitude by Country')
        plt.ylabel('Altitude (m)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Number of locations by country
        plt.subplot(2, 2, 4)
        plt.bar(countries, location_count, color='violet')
        plt.title('Number of Locations by Country')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(world_dir, 'country_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created country comparison figure in {world_dir}/country_comparison.png")
    else:
        logger.info("Not enough countries to create comparison figures (need at least 2)")
    
    return countries

def print_country_stats(country, country_data):
    """Print statistics for a country"""
    country_name = os.path.basename(country)
    
    if country_name in country_data and country_data[country_name]:
        locations = country_data[country_name]
        avg_survival = sum(r['survival_days'] for r in locations) / len(locations)
        max_icev = max(r['iceV_max'] for r in locations)
        avg_alt = sum(r['altitude'] for r in locations) / len(locations)
        
        print(f"  Locations processed: {len(locations)}")
        print(f"  Average survival days: {avg_survival:.2f}")
        print(f"  Maximum ice volume: {max_icev:.2f} m³")
        print(f"  Average altitude: {avg_alt:.2f} m")
        
        world_dir = os.path.join(dirname, 'data', 'world')
        safe_filename = sanitize_country_name(country_name)
        print(f"\nResults saved to: {os.path.join(world_dir, f'{safe_filename}.json')}")
        return True
    else:
        return False

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    st = time.time()

    args = parse_args()
    
    # Check if we should list countries and exit
    if args.list_countries:
        list_available_countries()
        sys.exit(0)
    
    # Build data path from country
    data_base_dir = os.path.join(dirname, 'data')
    country_dir = os.path.join(data_base_dir, args.country)
    if not os.path.exists(country_dir):
        print(f"Error: Country directory not found: {country_dir}")
        print("Run with --list_countries to see all available countries")
        sys.exit(1)
    
    # Get all filenames from data directory for this country
    filenames, data_dir = get_data_filenames(args.country)
    logger.info(f"Found {len(filenames)} CSV files to process for {args.country}")

    # Process all locations in parallel
    process_locations_parallel(filenames, args)
    
    # Consolidate results for this country
    logger.info(f"Consolidating results for {args.country}...")
    country_data = consolidate_results_parallel(filenames, data_dir)
    
    # Update the world summary file
    all_countries = update_world_summary()
    
    # # Create summary figures comparing countries if we have data for multiple countries
    # if len(all_countries) > 1:
    #     logger.info("Creating country comparison figures...")
    #     create_country_summary_figures()
    
    # Print summary of results
    print(f"\nProcessed results for country: {args.country}")
    print_country_stats(args.country, country_data)
    
    # Print summary of all countries
    print(f"\nTotal countries in world directory: {len(all_countries)}")
    total_locations = sum(data['location_count'] for data in all_countries.values())
    print(f"Total locations across all countries: {total_locations}")
    
    # Print execution time
    elapsed_time = (time.time() - st)/60
    print(f'\n\tTotal execution time: {round(elapsed_time,2)} min\n')
