@extends('layouts.main')

@section('content')
<div class="container-fluid p-4">
    <!-- Hero Section (Admin) -->
    <div class="card border-0 mb-5 shadow-lg overflow-hidden" 
         style="border-radius: 20px; background: linear-gradient(135deg, #43a047, #2e7d32); color: white;">
        <div class="card-body p-4 p-lg-5 d-flex flex-column flex-md-row justify-content-between align-items-center gap-3">
            <div class="text-center text-md-start">
                <h1 class="display-6 fw-bold mb-2">
                    <i class="bi bi-shield-lock me-2 text-white-50"></i>Admin Console
                </h1>
                <p class="lead mb-0 opacity-75">
                    Monitoring Kesehatan Sistem & Aktivitas User
                </p>
            </div>
            <div>
                <button class="btn btn-light text-success fw-bold btn-lg rounded-pill px-4 shadow-sm hover-scale" onclick="window.location.reload()">
                    <i class="bi bi-arrow-clockwise me-2"></i> Live Refresh
                </button>

            </div>
        </div>
    </div>

    <!-- System Health Stats (ETL + Realtime) -->
    <div class="row g-4 mb-4">
        <!-- GMV (Gross Merchandise Value) -->
        <div class="col-12 col-sm-6 col-xl-3">
            <div class="card h-100 border-0 shadow-sm overflow-hidden stat-card">
                <div class="card-body position-relative">
                    <div class="position-absolute top-0 end-0 mt-3 me-3 opacity-10">
                        <i class="bi bi-currency-dollar display-4 text-success"></i>
                    </div>
                    <p class="text-uppercase small fw-bold text-muted mb-1">Total GMV (System)</p>
                    <h3 class="fw-bold text-dark mb-1">Rp {{ number_format($adminStats['gmv'] ?? 0, 0, ',', '.') }}</h3>
                    <small class="text-success fw-bold">
                        <i class="bi bi-graph-up-arrow me-1"></i>
                        Perputaran Uang
                    </small>
                </div>
                <div class="progress" style="height: 4px;">
                    <div class="progress-bar bg-success" style="width: 100%"></div>
                </div>
            </div>
        </div>

        <!-- Total User Base -->
        <div class="col-12 col-sm-6 col-xl-3">
            <div class="card h-100 border-0 shadow-sm overflow-hidden stat-card">
                <div class="card-body position-relative">
                    <div class="position-absolute top-0 end-0 mt-3 me-3 opacity-10">
                        <i class="bi bi-people display-4 text-primary"></i>
                    </div>
                    <p class="text-uppercase small fw-bold text-muted mb-1">Total Pengguna</p>
                    <h3 class="fw-bold text-dark mb-1">{{ number_format($adminStats['total_users'] ?? 0) }}</h3>
                    <small class="text-primary fw-bold">
                        <span class="badge bg-primary-subtle text-primary rounded-pill">+{{ $adminStats['new_users_today'] ?? 0 }} Hari Ini</span>
                    </small>
                </div>
                <div class="progress" style="height: 4px;">
                    <div class="progress-bar bg-primary" style="width: 100%"></div>
                </div>
            </div>
        </div>

        <!-- Daily Transactions -->
        <div class="col-12 col-sm-6 col-xl-3">
            <div class="card h-100 border-0 shadow-sm overflow-hidden stat-card">
                <div class="card-body position-relative">
                    <div class="position-absolute top-0 end-0 mt-3 me-3 opacity-10">
                        <i class="bi bi-receipt display-4 text-info"></i>
                    </div>
                    <p class="text-uppercase small fw-bold text-muted mb-1">Transaksi Hari Ini</p>
                    <h3 class="fw-bold text-dark mb-1">{{ number_format($adminStats['total_tx_today'] ?? 0) }}</h3>
                    <small class="text-info fw-bold">
                        Activity Feed
                    </small>
                </div>
                <div class="progress" style="height: 4px;">
                    <div class="progress-bar bg-info" style="width: 100%"></div>
                </div>
            </div>
        </div>

        <!-- Pending Disputes/Nego -->
        <div class="col-12 col-sm-6 col-xl-3">
            <div class="card h-100 border-0 shadow-sm overflow-hidden stat-card">
                <div class="card-body position-relative">
                    <div class="position-absolute top-0 end-0 mt-3 me-3 opacity-10">
                        <i class="bi bi-exclamation-triangle display-4 text-warning"></i>
                    </div>
                    <p class="text-uppercase small fw-bold text-muted mb-1">Negosiasi Menunggu</p>
                    <h3 class="fw-bold text-dark mb-1">{{ number_format($adminStats['pending_nego'] ?? 0) }}</h3>
                    <small class="text-warning fw-bold">
                        Potential bottlenecks
                    </small>
                </div>
                <div class="progress" style="height: 4px;">
                    <div class="progress-bar bg-warning" style="width: 50%"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Analytics Charts Row -->
    <div class="row g-4 mb-4">
        <!-- GMV Trend Chart -->
        <div class="col-lg-8">
            <div class="card border-0 shadow-sm rounded-4 h-100">
                <div class="card-header bg-white border-0 p-4 pb-0 d-flex justify-content-between align-items-center">
                    <h5 class="fw-bold mb-0">Tren Gross Merchandise Value (GMV)</h5>
                    <div class="btn-group" role="group" aria-label="Time Filter">
                        <button type="button" class="btn btn-outline-secondary btn-sm filter-btn active" data-range="30d" onclick="updateChartFilter(this, '30d')">30 Hari</button>
                         <button type="button" class="btn btn-outline-secondary btn-sm filter-btn" data-range="24h" onclick="updateChartFilter(this, '24h')">24 Jam</button>
                        <button type="button" class="btn btn-outline-secondary btn-sm filter-btn" data-range="4w" onclick="updateChartFilter(this, '4w')">4 Minggu</button>
                        <button type="button" class="btn btn-outline-secondary btn-sm filter-btn" data-range="12m" onclick="updateChartFilter(this, '12m')">12 Bulan</button>
                    </div>
                </div>
                <div class="card-body p-4">
                    <div id="gmvChart" style="min-height: 350px;"></div>
                </div>
            </div>
        </div>

        <!-- Transaction Distribution -->
        <div class="col-lg-4">
            <div class="card border-0 shadow-sm rounded-4 h-100">
                <div class="card-header bg-white border-0 p-4 pb-0">
                    <h5 class="fw-bold mb-0">Status Transaksi</h5>
                </div>
                <div class="card-body p-4 d-flex align-items-center justify-content-center">
                    <div id="statusChart" class="w-100"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content Row -->
    <div class="row g-4">
        <!-- User Management (Quick View) -->
        <div class="col-lg-8">
            <div class="card border-0 shadow-sm rounded-4 h-100">
                <div class="card-header bg-white border-0 p-4 pb-0 d-flex justify-content-between align-items-center">
                    <h5 class="fw-bold mb-0">Manajemen Pengguna Terbaru</h5>
                    <a href="{{ route('admin.users.index') }}" class="btn btn-outline-success btn-sm rounded-pill px-3">
                        <i class="bi bi-people-fill me-1"></i> Lihat Semua
                    </a>
                </div>
                <div class="card-body p-4">
                    <div class="table-responsive">
                        <table class="table align-middle mb-0">
                            <thead class="bg-light text-uppercase text-secondary fw-bold" style="font-size: 0.85rem; border-bottom: 2px solid #e9ecef;">
                                <tr>
                                    <th class="ps-4 py-3 border-0 rounded-start">User Info</th>
                                    <th class="py-3 border-0">Role</th>
                                    <th class="py-3 border-0">Joined Date</th>
                                    <th class="pe-4 py-3 border-0 text-end rounded-end">Action</th>
                                </tr>
                            </thead>
                            <tbody style="font-size: 1rem;">
                                @forelse($latestUsers ?? [] as $user)
                                    <tr class="border-bottom hover-shadow-sm transition-all">
                                        <td class="ps-4 py-3 border-0">
                                            <div class="d-flex align-items-center">
                                                <div class="position-relative">
                                                    <div class="bg-success-subtle text-success rounded-circle d-flex align-items-center justify-content-center shadow-sm" style="width: 48px; height: 48px;">
                                                        <i class="bi bi-person-fill fs-4"></i>
                                                    </div>
                                                    <span class="position-absolute bottom-0 end-0 p-1 bg-success border border-light rounded-circle">
                                                        <span class="visually-hidden">Active</span>
                                                    </span>
                                                </div>
                                                <div class="ms-3">
                                                    <h6 class="mb-1 fw-bold text-dark">{{ $user->nama ?? $user->name ?? 'No Name' }}</h6>
                                                    <span class="d-block text-secondary small">{{ $user->email }}</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td class="py-3 border-0">
                                            @php
                                                $roleClasses = match($user->peran) {
                                                    'admin' => 'bg-dark text-white',
                                                    'petani' => 'bg-success text-white',
                                                    'pengepul' => 'bg-warning text-dark',
                                                    default => 'bg-secondary text-white'
                                                };
                                            @endphp
                                            <span class="badge {{ $roleClasses }} rounded-pill px-3 py-2 fw-normal shadow-sm">
                                                {{ ucfirst($user->peran) }}
                                            </span>
                                        </td>
                                        <td class="py-3 border-0 text-secondary">
                                            {{ $user->created_at->translatedFormat('d M Y') }}
                                            <small class="d-block text-muted" style="font-size: 0.75rem;">{{ $user->created_at->format('H:i') }}</small>
                                        </td>
                                        <td class="pe-4 py-3 border-0 text-end">
                                            <button class="btn btn-light text-success btn-sm rounded-circle shadow-sm" data-bs-toggle="tooltip" title="Edit details">
                                                <i class="bi bi-pencil-fill"></i>
                                            </button>
                                        </td>
                                    </tr>
                                @empty
                                    <tr>
                                        <td colspan="4" class="text-center py-5 text-muted bg-light rounded-3">
                                            <i class="bi bi-people display-4 opacity-50 mb-3 d-block"></i>
                                            Belum ada data pengguna.
                                        </td>
                                    </tr>
                                @endforelse
                            </tbody>
                        </table>
                        <div class="text-center mt-3 p-3 bg-light rounded-3">
                             <small class="text-muted">Fitur manajemen user lengkap tersedia di menu "Users"</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Links (Control Panel) -->
        <div class="col-lg-4">
            <div class="card border-0 shadow-sm rounded-4 h-100">
                 <div class="card-header bg-white border-bottom-0 p-4 pb-0">
                    <h5 class="fw-bold mb-0">Control Panel</h5>
                </div>
                <div class="card-body p-4">
                     <div class="d-grid gap-3">
                        <a href="{{ route('admin.backup') }}" class="btn btn-light text-start p-3 border-0 shadow-sm rounded-4 d-flex align-items-center hover-scale">
                            <div class="bg-success-subtle text-success rounded-circle p-3 me-3">
                                <i class="bi bi-database-check fs-4"></i>
                            </div>
                            <div>
                                <div class="fw-bold text-dark h6 mb-0">Database Backup</div>
                                <small class="text-secondary">Download JSON Dump</small>
                            </div>
                        </a>

                        <a href="{{ route('dashboard.data') }}" target="_blank" class="btn btn-light text-start p-3 border-0 shadow-sm rounded-4 d-flex align-items-center hover-scale">
                            <div class="bg-primary-subtle text-primary rounded-circle p-3 me-3">
                                <i class="bi bi-braces fs-4"></i>
                            </div>
                            <div>
                                <div class="fw-bold text-dark h6 mb-0">Raw JSON API</div>
                                <small class="text-secondary">Debug dashboard data</small>
                            </div>
                        </a>
                        
                        <a href="{{ route('admin.logs') }}" target="_blank" class="btn btn-light text-start p-3 border-0 shadow-sm rounded-4 d-flex align-items-center hover-scale">
                            <div class="bg-warning-subtle text-warning rounded-circle p-3 me-3">
                                <i class="bi bi-terminal fs-4"></i>
                            </div>
                            <div>
                                <div class="fw-bold text-dark h6 mb-0">System Logs</div>
                                <small class="text-secondary">View laravel.log</small>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    .stat-card { transition: transform 0.2s; }
    .stat-card:hover { transform: translateY(-5px); }
    .hover-shadow:hover { background-color: #f8f9fa; border-color: transparent; box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
    .bg-primary-subtle { background-color: rgba(13, 110, 253, 0.1) !important; }
    .bg-success-subtle { background-color: rgba(25, 135, 84, 0.1) !important; }
    .hover-bg-light:hover { background-color: #f8f9fa; }
</style>
<script>
    let gmvChart; 

    document.addEventListener('DOMContentLoaded', function() {
        const chartData = @json($chartData ?? []);

        // 1. GMV Line Chart
        if (chartData.trend_labels) {
            const gmvOptions = {
                series: [{
                    name: 'GMV (Rp)',
                    data: chartData.trend_gmv
                }],
                chart: {
                    type: 'area', // Area for nice gradient look
                    height: 350,
                    toolbar: { show: false },
                    fontFamily: 'Inter, sans-serif'
                },
                colors: ['#2E7D32'], // Official Green
                stroke: { curve: 'smooth', width: 2 },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shadeIntensity: 1,
                        opacityFrom: 0.7,
                        opacityTo: 0.1,
                    }
                },
                xaxis: {
                    categories: chartData.trend_labels,
                    axisBorder: { show: false },
                    axisTicks: { show: false }
                },
                yaxis: {
                    labels: {
                        formatter: function (value) {
                            return new Intl.NumberFormat('id-ID', { notation: "compact" }).format(value);
                        }
                    }
                },
                grid: {
                    strokeDashArray: 4,
                },
                tooltip: { y: { formatter: function (val) { return "Rp " + new Intl.NumberFormat('id-ID').format(val) } } }
            };

            gmvChart = new ApexCharts(document.querySelector("#gmvChart"), gmvOptions);
            gmvChart.render();
        }

        // 2. Status Donut Chart
        if (chartData.status_distribution) {
            // Transform associative array to arrays
            const labels = Object.keys(chartData.status_distribution);
            const series = Object.values(chartData.status_distribution);

            const statusOptions = {
                series: series,
                labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)), 
                chart: {
                    type: 'donut',
                    height: 350,
                    fontFamily: 'Inter, sans-serif'
                },
                colors: ['#FFC107', '#4CAF50', '#2196F3', '#FF5722', '#9E9E9E'], // Colors mapping
                plotOptions: {
                    pie: {
                        donut: {
                            labels: {
                                show: true,
                                total: {
                                    show: true,
                                    label: 'Total',
                                    formatter: function (w) {
                                        return w.globals.seriesTotals.reduce((a, b) => a + b, 0)
                                    }
                                }
                            }
                        }
                    }
                },
                legend: { position: 'bottom' }
            };

            const statusChart = new ApexCharts(document.querySelector("#statusChart"), statusOptions);
            statusChart.render();
        }
    });
    
    async function updateChartFilter(btn, range) {
        // UI Update
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active', 'btn-secondary'));
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.add('btn-outline-secondary'));
        
        document.querySelectorAll(`button[data-range="${range}"]`).forEach(b => {
             b.classList.add('active', 'btn-secondary');
             b.classList.remove('btn-outline-secondary');
        });

        // Fetch Data
        try {
            const response = await fetch(`{{ route('dashboard.chart-data') }}?range=${range}`);
            const data = await response.json();

            if (gmvChart && data.trend_labels) {
                gmvChart.updateOptions({
                    xaxis: { categories: data.trend_labels }
                });
                gmvChart.updateSeries([{
                    name: 'GMV (Rp)',
                    data: data.trend_gmv
                }]);
            }
        } catch (error) {
            console.error('Error fetching admin chart data:', error);
            alert('Gagal memuat data grafik admin.');
        }
    }
</script>
@endsection
